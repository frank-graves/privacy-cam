use anyhow::Result;
use image::{imageops::{resize, FilterType}, RgbImage};
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// YOLOv8 exige que el tamaño de entrada sea múltiplo de 32.
const STRIDE: u32 = 32;
/// Gris que usa Ultralytics para el relleno del letterbox (114/255).
const PAD_VALUE: f32 = 114.0 / 255.0;

/// Caja en coordenadas de la imagen ORIGINAL (píxeles). Sin márgenes extra:
/// el margen de seguridad lo aplica quien censura.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub confidence: f32,
}

impl BoundingBox {
    pub fn area(&self) -> f32 {
        self.width.max(0.0) * self.height.max(0.0)
    }

    /// Intersection over Union con otra caja (0.0 si no se tocan).
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }
}

/// Parámetros del detector.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Lado LARGO de la entrada en píxeles (múltiplo de 32). La imagen se
    /// escala manteniendo proporción: 640x480 con 320 -> entrada de 320x256.
    /// Medido con este modelo en un frame de 640x480: a 320 detecta caras de
    /// hasta ~45 px de ancho, a 416 ~41 px, a 640 ~35 px. Una cara de webcam
    /// normal mide 120-250 px, así que 320 alcanza y es ~2x más barato que 416.
    /// Subilo (416/480) solo si necesitás censurar gente MUY lejos de la cámara.
    pub input_long_side: u32,
    /// Confianza mínima. Para privacidad conviene ser permisivo: un falso
    /// positivo pixela de más, un falso negativo deja una cara a la vista.
    pub conf_threshold: f32,
    /// IoU a partir del cual dos cajas se consideran la misma cara.
    pub nms_iou: f32,
    /// Máximo de caras devueltas.
    pub max_faces: usize,
    /// Hilos de ORT. Por defecto deja un núcleo libre para captura/UI.
    pub intra_threads: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        Self {
            input_long_side: 320,
            conf_threshold: 0.25,
            nms_iou: 0.45,
            max_faces: 8,
            intra_threads: cores.saturating_sub(1).clamp(1, 2),
        }
    }
}

pub struct FaceDetector {
    session: Mutex<Session>,
    config: DetectorConfig,
}

impl FaceDetector {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::with_config(model_path, DetectorConfig::default())
    }

    pub fn with_config<P: AsRef<Path>>(model_path: P, config: DetectorConfig) -> Result<Self> {
        let environment = ort::init()
            .with_name("privacy-cam")
            .build()
            .map_err(|e| anyhow::anyhow!("Error creando entorno ORT: {}", e))?;

        let session = Session::builder(&environment)
            .map_err(|e| anyhow::anyhow!("Error creando session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Error en optimización: {}", e))?
            .with_intra_threads(config.intra_threads)
            .map_err(|e| anyhow::anyhow!("Error configurando hilos: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Error cargando el modelo ONNX: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            config,
        })
    }

    /// Corre una inferencia en vacío. La primera inferencia de ORT es bastante
    /// más lenta (reserva memoria, planifica el grafo) y dispararía el fail-safe.
    /// Devuelve los milisegundos que tardó.
    pub fn warm_up(&self, width: u32, height: u32) -> Result<u128> {
        let start = Instant::now();
        let blank = RgbImage::new(width, height);
        self.detect(&blank)?;
        self.detect(&blank)?;
        Ok(start.elapsed().as_millis() / 2)
    }

    pub fn detect(&self, img: &RgbImage) -> Result<Vec<BoundingBox>> {
        let lb = compute_letterbox(img.width(), img.height(), self.config.input_long_side);
        let input_data = preprocess(img, &lb);

        let input_tensor =
            Tensor::from_array(([1i64, 3, lb.in_h as i64, lb.in_w as i64], input_data))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("No se pudo adquirir el lock de la sesión"))?;

        let outputs = session.run(inputs![input_tensor])?;

        let extracted = outputs[0].try_extract_tensor::<f32>()?;
        let shape = extracted.0;
        let data = extracted.1;

        if shape.len() != 3 || shape[0] != 1 || shape[1] < 5 {
            return Err(anyhow::anyhow!("Formato de salida inesperado: {:?}", shape));
        }

        let num_anchors = shape[2] as usize;
        if data.len() < 5 * num_anchors {
            return Err(anyhow::anyhow!("Salida truncada: {} valores", data.len()));
        }

        let candidates = decode_output(
            data,
            num_anchors,
            &lb,
            img.width() as f32,
            img.height() as f32,
            self.config.conf_threshold,
        );

        Ok(nms(candidates, self.config.nms_iou, self.config.max_faces))
    }
}

// ---------------------------------------------------------------------------
// Pre/post-procesado (funciones puras, sin ORT, para poder testearlas)
// ---------------------------------------------------------------------------

/// Cómo se acomoda la imagen original dentro del tensor de entrada.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// Tamaño final del tensor (múltiplos de 32).
    pub in_w: u32,
    pub in_h: u32,
    /// Tamaño de la imagen reescalada dentro del tensor.
    pub new_w: u32,
    pub new_h: u32,
    /// Relleno a izquierda y arriba.
    pub pad_x: u32,
    pub pad_y: u32,
    /// Factor original -> tensor.
    pub scale: f32,
}

fn round_up(v: u32, multiple: u32) -> u32 {
    ((v + multiple - 1) / multiple) * multiple
}

/// Escala manteniendo la proporción y rellena SOLO lo mínimo para llegar a un
/// múltiplo de 32 (el modelo tiene ejes dinámicos, no hace falta un cuadrado).
pub fn compute_letterbox(img_w: u32, img_h: u32, long_side: u32) -> Letterbox {
    let long = img_w.max(img_h).max(1) as f32;
    let scale = long_side as f32 / long;
    let new_w = ((img_w as f32 * scale).round() as u32).max(1);
    let new_h = ((img_h as f32 * scale).round() as u32).max(1);
    let in_w = round_up(new_w, STRIDE);
    let in_h = round_up(new_h, STRIDE);
    Letterbox {
        in_w,
        in_h,
        new_w,
        new_h,
        pad_x: (in_w - new_w) / 2,
        pad_y: (in_h - new_h) / 2,
        scale,
    }
}

/// RGB u8 -> tensor f32 planar (CHW) normalizado a [0,1], con letterbox.
fn preprocess(img: &RgbImage, lb: &Letterbox) -> Vec<f32> {
    let in_w = lb.in_w as usize;
    let in_h = lb.in_h as usize;
    let plane = in_w * in_h;
    let mut data = vec![PAD_VALUE; 3 * plane];

    let resized;
    let src: &[u8] = if lb.new_w == img.width() && lb.new_h == img.height() {
        img.as_raw().as_slice()
    } else {
        resized = resize(img, lb.new_w, lb.new_h, FilterType::Triangle);
        resized.as_raw().as_slice()
    };

    let new_w = lb.new_w as usize;
    let k = 1.0 / 255.0;
    for y in 0..lb.new_h as usize {
        let dst_row = (y + lb.pad_y as usize) * in_w + lb.pad_x as usize;
        let src_row = y * new_w * 3;
        for x in 0..new_w {
            let s = src_row + x * 3;
            let d = dst_row + x;
            data[d] = src[s] as f32 * k;
            data[plane + d] = src[s + 1] as f32 * k;
            data[2 * plane + d] = src[s + 2] as f32 * k;
        }
    }
    data
}

/// Convierte la salida de YOLOv8 (layout [5, N]: cx, cy, w, h, conf) a cajas en
/// coordenadas de la imagen original, deshaciendo el letterbox.
pub fn decode_output(
    data: &[f32],
    num_anchors: usize,
    lb: &Letterbox,
    img_w: f32,
    img_h: f32,
    conf_threshold: f32,
) -> Vec<BoundingBox> {
    let pad_x = lb.pad_x as f32;
    let pad_y = lb.pad_y as f32;
    let mut out = Vec::new();

    for i in 0..num_anchors {
        let conf = data[4 * num_anchors + i];
        // `!(a > b)` también descarta NaN.
        if !(conf > conf_threshold) {
            continue;
        }
        let cx = data[i];
        let cy = data[num_anchors + i];
        let w = data[2 * num_anchors + i];
        let h = data[3 * num_anchors + i];

        let x1 = ((cx - w / 2.0 - pad_x) / lb.scale).clamp(0.0, img_w);
        let y1 = ((cy - h / 2.0 - pad_y) / lb.scale).clamp(0.0, img_h);
        let x2 = ((cx + w / 2.0 - pad_x) / lb.scale).clamp(0.0, img_w);
        let y2 = ((cy + h / 2.0 - pad_y) / lb.scale).clamp(0.0, img_h);

        if x2 - x1 < 2.0 || y2 - y1 < 2.0 {
            continue;
        }
        out.push(BoundingBox {
            x: x1,
            y: y1,
            width: x2 - x1,
            height: y2 - y1,
            confidence: conf,
        });
    }
    out
}

/// Non-Maximum Suppression: de varias cajas sobre la misma cara deja la de
/// mayor confianza. Sin esto, `take(3)` devolvía tres cajas de UNA sola cara.
pub fn nms(mut boxes: Vec<BoundingBox>, iou_threshold: f32, max_keep: usize) -> Vec<BoundingBox> {
    boxes.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<BoundingBox> = Vec::new();
    for b in boxes {
        if kept.len() >= max_keep {
            break;
        }
        if kept.iter().all(|k| k.iou(&b) < iou_threshold) {
            kept.push(b);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_640x480_is_rectangular() {
        let lb = compute_letterbox(640, 480, 416);
        assert_eq!((lb.new_w, lb.new_h), (416, 312));
        assert_eq!((lb.in_w, lb.in_h), (416, 320));
        assert_eq!((lb.pad_x, lb.pad_y), (0, 4));
    }

    #[test]
    fn letterbox_portrait_and_odd_sizes() {
        let lb = compute_letterbox(480, 640, 416);
        assert_eq!((lb.in_w, lb.in_h), (320, 416));
        let lb = compute_letterbox(1, 1, 416);
        assert!(lb.in_w % 32 == 0 && lb.in_h % 32 == 0 && lb.in_w >= 32);
    }

    #[test]
    fn preprocess_places_pixels_and_padding() {
        let mut img = RgbImage::new(640, 480);
        for p in img.pixels_mut() {
            *p = image::Rgb([255, 0, 0]);
        }
        let lb = compute_letterbox(640, 480, 416);
        let d = preprocess(&img, &lb);
        let plane = (lb.in_w * lb.in_h) as usize;
        assert_eq!(d.len(), 3 * plane);
        // fila de relleno (arriba): gris
        assert!((d[0] - PAD_VALUE).abs() < 1e-6);
        // centro: rojo puro
        let c = (lb.in_h as usize / 2) * lb.in_w as usize + lb.in_w as usize / 2;
        assert!(d[c] > 0.99 && d[plane + c] < 0.01 && d[2 * plane + c] < 0.01);
    }

    #[test]
    fn decode_undoes_letterbox() {
        // Cara de 100x100 centrada en (320,240) de una imagen 640x480.
        let lb = compute_letterbox(640, 480, 416);
        let n = 1;
        let s = lb.scale;
        let data = vec![
            320.0 * s + lb.pad_x as f32, // cx
            240.0 * s + lb.pad_y as f32, // cy
            100.0 * s,                   // w
            100.0 * s,                   // h
            0.9,                         // conf
        ];
        let b = decode_output(&data, n, &lb, 640.0, 480.0, 0.25);
        assert_eq!(b.len(), 1);
        assert!((b[0].x - 270.0).abs() < 1.0 && (b[0].y - 190.0).abs() < 1.0);
        assert!((b[0].width - 100.0).abs() < 1.0 && (b[0].height - 100.0).abs() < 1.0);
    }

    #[test]
    fn nms_collapses_duplicates_but_keeps_other_faces() {
        let mk = |x: f32, c: f32| BoundingBox { x, y: 10.0, width: 100.0, height: 100.0, confidence: c };
        // 3 cajas casi iguales (misma cara) + 1 cara lejana.
        let boxes = vec![mk(100.0, 0.6), mk(104.0, 0.9), mk(98.0, 0.7), mk(400.0, 0.5)];
        let out = nms(boxes, 0.45, 8);
        assert_eq!(out.len(), 2);
        assert!((out[0].confidence - 0.9).abs() < 1e-6);
        assert!((out[1].x - 400.0).abs() < 1e-6);
    }

    #[test]
    fn nms_survives_nan_and_respects_max() {
        let mk = |x: f32, c: f32| BoundingBox { x, y: 0.0, width: 10.0, height: 10.0, confidence: c };
        let out = nms(vec![mk(0.0, f32::NAN), mk(50.0, 0.8), mk(100.0, 0.7)], 0.45, 2);
        assert_eq!(out.len(), 2);
    }
}
