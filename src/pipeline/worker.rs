//! Worker = "compositor" + hilo detector.
//!
//! Antes: un solo hilo hacía `recibir frame -> detectar (lento) -> pixelar -> enviar`,
//! así que los FPS de la cámara dependían de la IA y, si la IA no daba abasto,
//! la UI se quedaba con el último frame (parecía "congelado").
//!
//! Ahora hay dos hilos:
//!   * Detector: toma SIEMPRE el frame más nuevo que le dejaron y calcula cajas.
//!   * Compositor (este worker): por cada frame de la cámara aplica las últimas
//!     cajas conocidas y lo envía. La fluidez ya no depende de la IA.
//!
//! Reglas de privacidad:
//!   * Hasta que no haya una primera detección, NO se muestra nada (nunca un
//!     frame crudo).
//!   * Una cara detectada sigue censurada `hold` ms aunque el detector falle un
//!     frame (no hay "parpadeos" donde se vea la cara).
//!   * Si el detector deja de responder (se colgó, CPU saturada) -> Freeze.
//!     "No hay caras" y "el detector no responde" son cosas distintas.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use image::RgbImage;
use log::{error, info, warn};

use crate::inference::detector::{BoundingBox, FaceDetector};
use crate::pipeline::channels::{
    CaptureMsg, DetectionJob, DetectionResult, InferenceMsg, LatestSlot, VideoFrame,
};
use crate::state::{AppState, SharedState};

/// Parámetros ajustables del pipeline.
#[derive(Debug, Clone)]
pub struct PipelineTuning {
    /// Se manda un frame al detector cada N frames de cámara. Más alto = menos
    /// CPU. A 30 fps, N=3 son ~10 detecciones/s.
    pub detect_every_n: u64,
    /// Cuánto tiempo sigue censurada una cara después de la última vez que se
    /// la vio.
    pub hold: Duration,
    /// Margen alrededor de la caja, proporcional a su tamaño (0.25 = 25 % por lado).
    pub margin_ratio: f32,
    /// Margen mínimo en píxeles.
    pub min_margin_px: f32,
    /// Peso de la caja nueva al suavizar con la anterior (1.0 = sin suavizado).
    pub smoothing: f32,
    /// IoU mínimo para considerar que una detección es la misma cara que un track.
    pub match_iou: f32,
    /// Piso del timeout del fail-safe. `fail_safe_timeout_ms` de la config puede
    /// ser muy bajo (200 ms) para PCs lentas: con detección cada N frames
    /// congelaría la imagen a cada rato.
    pub min_stall: Duration,
}

impl Default for PipelineTuning {
    fn default() -> Self {
        Self {
            detect_every_n: 3,
            hold: Duration::from_millis(600),
            margin_ratio: 0.25,
            min_margin_px: 6.0,
            smoothing: 0.7,
            match_iou: 0.15,
            min_stall: Duration::from_millis(500),
        }
    }
}

/// Misma firma que antes: `main.rs` no necesita cambios.
pub fn spawn_worker_thread(
    detector: FaceDetector,
    capture_rx: Receiver<CaptureMsg>,
    inference_tx: Sender<InferenceMsg>,
    state: SharedState,
) -> thread::JoinHandle<()> {
    spawn_worker_thread_with(detector, capture_rx, inference_tx, state, PipelineTuning::default())
}

pub fn spawn_worker_thread_with(
    detector: FaceDetector,
    capture_rx: Receiver<CaptureMsg>,
    inference_tx: Sender<InferenceMsg>,
    state: SharedState,
    tuning: PipelineTuning,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        info!("Hilo Worker (compositor) iniciado.");

        let jobs: LatestSlot<DetectionJob> = LatestSlot::new();
        let results: LatestSlot<DetectionResult> = LatestSlot::new();

        let det_handle = spawn_detection_thread(detector, jobs.clone(), results.clone(), state.clone());

        run_compositor(&capture_rx, &inference_tx, &state, &tuning, &jobs, &results);

        jobs.close();
        results.close();
        let _ = det_handle.join();
        info!("Hilo Worker finalizado.");
    })
}

// ---------------------------------------------------------------------------
// Hilo detector
// ---------------------------------------------------------------------------

fn spawn_detection_thread(
    detector: FaceDetector,
    jobs: LatestSlot<DetectionJob>,
    results: LatestSlot<DetectionResult>,
    state: SharedState,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name("detector".into())
        .spawn(move || {
            match detector.warm_up(640, 480) {
                Ok(ms) => info!("Detector listo (~{} ms por inferencia).", ms),
                Err(e) => warn!("Falló el warm-up del detector: {}", e),
            }

            let mut consecutive_errors = 0u32;
            while state.is_running() {
                let Some(job) = jobs.wait_take(Duration::from_millis(100)) else {
                    if jobs.is_closed() {
                        break;
                    }
                    continue;
                };

                let start = Instant::now();
                match detector.detect(&job.image) {
                    Ok(boxes) => {
                        consecutive_errors = 0;
                        results.put(DetectionResult {
                            boxes,
                            captured_at: job.captured_at,
                            inference_ms: start.elapsed().as_millis(),
                        });
                    }
                    Err(e) => {
                        // Sin resultado nuevo, el compositor detecta el atraso y
                        // activa el fail-safe por sí solo.
                        consecutive_errors += 1;
                        if consecutive_errors == 1 || consecutive_errors % 30 == 0 {
                            error!("Error en la inferencia ({} seguidos): {}", consecutive_errors, e);
                        }
                    }
                }
            }
            info!("Hilo detector finalizado.");
        })
        .expect("No se pudo crear el hilo del detector")
}

// ---------------------------------------------------------------------------
// Compositor
// ---------------------------------------------------------------------------

fn run_compositor(
    capture_rx: &Receiver<CaptureMsg>,
    inference_tx: &Sender<InferenceMsg>,
    state: &SharedState,
    tuning: &PipelineTuning,
    jobs: &LatestSlot<DetectionJob>,
    results: &LatestSlot<DetectionResult>,
) {
    let mut tracker = FaceTracker::new(tuning);
    // Momento de captura del frame de la detección más reciente que tenemos.
    let mut last_detection_frame: Option<Instant> = None;
    let mut last_inference_ms: u128 = 0;
    let mut frame_idx: u64 = 0;
    let mut stats = Stats::new();
    let started = Instant::now();
    let mut warned_slow_start = false;

    while state.is_running() {
        // 1. Esperar un frame (con timeout para poder ver si hay que apagar).
        let first = match capture_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };

        // 2. Si nos atrasamos, saltar al frame MÁS NUEVO y descartar los viejos.
        let mut latest: Option<VideoFrame> = None;
        let mut pending = Some(first);
        while let Some(msg) = pending.take().or_else(|| capture_rx.try_recv().ok()) {
            match msg {
                CaptureMsg::NewFrame(f) => {
                    if latest.replace(f).is_some() {
                        stats.skipped += 1;
                    }
                }
                CaptureMsg::Error(e) => error!("Error de captura: {}", e),
            }
        }
        let Some(frame) = latest else { continue };

        // 3. Recoger la detección más reciente (si hay).
        if let Some(res) = results.take() {
            tracker.update(&res.boxes, res.captured_at);
            if last_detection_frame.map_or(true, |t| res.captured_at > t) {
                last_detection_frame = Some(res.captured_at);
            }
            last_inference_ms = res.inference_ms;
        }

        // 4. Estado global de la app.
        let app_state = state.get_state();
        match app_state {
            AppState::Shutdown => break,
            AppState::Paused => {
                // Al reanudar, esperar una detección fresca antes de mostrar.
                last_detection_frame = None;
                continue;
            }
            _ => {}
        }

        // 5. Ofrecer este frame al detector (sobrescribe el anterior si no lo tomó).
        if frame_idx % tuning.detect_every_n.max(1) == 0 {
            jobs.put(DetectionJob {
                image: Arc::clone(&frame.image),
                captured_at: frame.captured_at,
            });
        }
        frame_idx += 1;

        // 6. Fail-safe.
        let cfg = state.get_config();
        let timeout = Duration::from_millis(cfg.fail_safe_timeout_ms).max(tuning.min_stall);

        let Some(last_det) = last_detection_frame else {
            // Todavía no hay NINGUNA detección: no mostramos frames crudos.
            if !warned_slow_start && started.elapsed() > Duration::from_secs(10) {
                warn!("El detector lleva más de 10 s sin responder. ¿Modelo o CPU?");
                warned_slow_start = true;
            }
            continue;
        };

        let in_freeze = matches!(app_state, AppState::Freeze { .. });

        if last_det.elapsed() > timeout {
            if !in_freeze {
                let reason = format!(
                    "El detector no responde hace más de {} ms",
                    last_det.elapsed().as_millis()
                );
                warn!("FAIL-SAFE: {}", reason);
                state.set_state(AppState::Freeze { reason: reason.clone() });
                let _ = inference_tx.try_send(InferenceMsg::Failed(reason));
            }
            continue; // no publicamos: la salida queda congelada
        }

        if in_freeze {
            if cfg.require_manual_resume {
                continue; // la UI debe volver a Normal a mano
            }
            info!("Detector recuperado. Reanudando.");
            state.set_state(AppState::Normal);
        }

        // 7. Censurar con las cajas vigentes y publicar.
        tracker.prune(Instant::now());
        let boxes = tracker.boxes();

        let mut image = frame.image;
        {
            // Solo copia los píxeles si el detector todavía tiene este frame.
            let buf = Arc::make_mut(&mut image);
            for b in &boxes {
                censor_box(buf, b, tuning);
            }
        }

        let _ = inference_tx.try_send(InferenceMsg::Processed(VideoFrame {
            image,
            captured_at: frame.captured_at,
        }));

        stats.frames += 1;
        stats.report(last_inference_ms, boxes.len());
    }
}

// ---------------------------------------------------------------------------
// Seguimiento simple de caras (mantiene la censura entre detecciones)
// ---------------------------------------------------------------------------

struct Track {
    bbox: BoundingBox,
    /// Momento de captura del frame donde se la vio por última vez.
    last_seen: Instant,
}

pub struct FaceTracker {
    tracks: Vec<Track>,
    smoothing: f32,
    match_iou: f32,
    hold: Duration,
}

impl FaceTracker {
    pub fn new(t: &PipelineTuning) -> Self {
        Self {
            tracks: Vec::new(),
            smoothing: t.smoothing.clamp(0.0, 1.0),
            match_iou: t.match_iou,
            hold: t.hold,
        }
    }

    /// Incorpora las detecciones de un frame capturado en `seen_at`.
    pub fn update(&mut self, detections: &[BoundingBox], seen_at: Instant) {
        let mut used = vec![false; self.tracks.len()];
        let mut fresh = Vec::new();

        for det in detections {
            let mut best: Option<(usize, f32)> = None;
            for (i, t) in self.tracks.iter().enumerate() {
                if used[i] {
                    continue;
                }
                let iou = t.bbox.iou(det);
                if iou >= self.match_iou && best.map_or(true, |(_, b)| iou > b) {
                    best = Some((i, iou));
                }
            }
            match best {
                Some((i, _)) => {
                    used[i] = true;
                    let s = self.smoothing;
                    let t = &mut self.tracks[i];
                    t.bbox = blend(&t.bbox, det, s);
                    t.last_seen = seen_at;
                }
                None => fresh.push(Track { bbox: *det, last_seen: seen_at }),
            }
        }
        self.tracks.extend(fresh);
    }

    /// Descarta las caras que no se ven hace más de `hold`.
    pub fn prune(&mut self, now: Instant) {
        let hold = self.hold;
        self.tracks
            .retain(|t| now.saturating_duration_since(t.last_seen) <= hold);
    }

    pub fn boxes(&self) -> Vec<BoundingBox> {
        self.tracks.iter().map(|t| t.bbox).collect()
    }
}

fn blend(old: &BoundingBox, new: &BoundingBox, w: f32) -> BoundingBox {
    let l = |a: f32, b: f32| a + (b - a) * w;
    BoundingBox {
        x: l(old.x, new.x),
        y: l(old.y, new.y),
        width: l(old.width, new.width),
        height: l(old.height, new.height),
        confidence: new.confidence,
    }
}

// ---------------------------------------------------------------------------
// Censura
// ---------------------------------------------------------------------------

/// Expande la caja con el margen configurado y la pixela.
fn censor_box(img: &mut RgbImage, b: &BoundingBox, t: &PipelineTuning) {
    let (iw, ih) = (img.width() as f32, img.height() as f32);
    let mx = (b.width * t.margin_ratio).max(t.min_margin_px);
    let my = (b.height * t.margin_ratio).max(t.min_margin_px);

    let x0 = (b.x - mx).floor().max(0.0) as u32;
    let y0 = (b.y - my).floor().max(0.0) as u32;
    let x1 = (b.x + b.width + mx).ceil().min(iw) as u32;
    let y1 = (b.y + b.height + my).ceil().min(ih) as u32;

    if x1 <= x0 || y1 <= y0 {
        return;
    }
    // ~8 bloques a lo ancho: la cara es irreconocible sin importar su tamaño
    // (con bloques fijos de 20 px, una cara chica quedaba apenas difuminada).
    let block = ((x1 - x0).max(y1 - y0) / 8).clamp(6, 64);
    pixelate_region(img, x0, y0, x1, y1, block);
}

/// Pixelado in-place: promedia cada bloque y lo rellena. Sin crop/resize/overlay,
/// o sea sin alocar imágenes intermedias.
pub fn pixelate_region(img: &mut RgbImage, x0: u32, y0: u32, x1: u32, y1: u32, block: u32) {
    let w = img.width() as usize;
    let x1 = x1.min(img.width());
    let y1 = y1.min(img.height());
    let block = block.max(1);
    let raw: &mut [u8] = &mut **img;

    let mut by = y0;
    while by < y1 {
        let bh = block.min(y1 - by);
        let mut bx = x0;
        while bx < x1 {
            let bw = block.min(x1 - bx);

            let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
            for yy in by..by + bh {
                let row = (yy as usize * w + bx as usize) * 3;
                for px in raw[row..row + bw as usize * 3].chunks_exact(3) {
                    r += px[0] as u32;
                    g += px[1] as u32;
                    b += px[2] as u32;
                }
            }
            let n = bw * bh;
            let avg = [(r / n) as u8, (g / n) as u8, (b / n) as u8];

            for yy in by..by + bh {
                let row = (yy as usize * w + bx as usize) * 3;
                for px in raw[row..row + bw as usize * 3].chunks_exact_mut(3) {
                    px.copy_from_slice(&avg);
                }
            }
            bx += block;
        }
        by += block;
    }
}

// ---------------------------------------------------------------------------
// Estadísticas (un log cada 5 s en vez de uno por frame)
// ---------------------------------------------------------------------------

struct Stats {
    since: Instant,
    frames: u32,
    skipped: u32,
}

impl Stats {
    fn new() -> Self {
        Self { since: Instant::now(), frames: 0, skipped: 0 }
    }

    fn report(&mut self, inference_ms: u128, faces: usize) {
        let secs = self.since.elapsed().as_secs_f32();
        if secs >= 5.0 {
            info!(
                "{:.1} FPS | detección: {} ms | caras censuradas: {} | frames saltados: {}",
                self.frames as f32 / secs,
                inference_ms,
                faces,
                self.skipped
            );
            self.since = Instant::now();
            self.frames = 0;
            self.skipped = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(x: f32, y: f32, w: f32, h: f32) -> BoundingBox {
        BoundingBox { x, y, width: w, height: h, confidence: 0.9 }
    }

    #[test]
    fn tracker_holds_face_after_detector_misses_then_drops_it() {
        let t = PipelineTuning::default();
        let mut tr = FaceTracker::new(&t);
        let t0 = Instant::now();

        tr.update(&[bb(100.0, 100.0, 80.0, 80.0)], t0);
        // el detector no ve nada en los siguientes frames...
        tr.prune(t0 + Duration::from_millis(300));
        assert_eq!(tr.boxes().len(), 1, "debe seguir censurada dentro del hold");
        // ...pero pasado el hold, se descarta.
        tr.prune(t0 + Duration::from_millis(900));
        assert_eq!(tr.boxes().len(), 0);
    }

    #[test]
    fn tracker_matches_same_face_and_separates_different_ones() {
        let t = PipelineTuning::default();
        let mut tr = FaceTracker::new(&t);
        let t0 = Instant::now();

        tr.update(&[bb(100.0, 100.0, 80.0, 80.0), bb(400.0, 100.0, 80.0, 80.0)], t0);
        assert_eq!(tr.boxes().len(), 2);
        // misma gente, un poco movida: no deben duplicarse
        tr.update(&[bb(110.0, 100.0, 80.0, 80.0), bb(405.0, 102.0, 80.0, 80.0)], t0 + Duration::from_millis(100));
        assert_eq!(tr.boxes().len(), 2);
        // aparece una tercera persona
        tr.update(
            &[bb(110.0, 100.0, 80.0, 80.0), bb(405.0, 102.0, 80.0, 80.0), bb(250.0, 300.0, 60.0, 60.0)],
            t0 + Duration::from_millis(200),
        );
        assert_eq!(tr.boxes().len(), 3);
    }

    #[test]
    fn pixelate_only_touches_the_region_and_destroys_detail() {
        let mut img = RgbImage::new(100, 100);
        for (x, y, p) in img.enumerate_pixels_mut() {
            *p = image::Rgb([(x * 2) as u8, (y * 2) as u8, ((x + y) % 256) as u8]);
        }
        let orig = img.clone();
        pixelate_region(&mut img, 20, 20, 60, 60, 10);

        // fuera de la región: intacto
        assert_eq!(img.get_pixel(5, 5), orig.get_pixel(5, 5));
        assert_eq!(img.get_pixel(70, 70), orig.get_pixel(70, 70));
        // dentro de un bloque: todos los píxeles iguales
        assert_eq!(img.get_pixel(20, 20), img.get_pixel(29, 29));
        // y distinto del original
        assert_ne!(img.get_pixel(21, 25), orig.get_pixel(21, 25));
    }

    #[test]
    fn censor_box_clamps_to_image_borders() {
        let t = PipelineTuning::default();
        let mut img = RgbImage::new(64, 48);
        // caja que se sale por todos lados: no debe entrar en pánico
        censor_box(&mut img, &bb(-20.0, -20.0, 200.0, 200.0), &t);
        censor_box(&mut img, &bb(60.0, 45.0, 30.0, 30.0), &t);
        censor_box(&mut img, &bb(500.0, 500.0, 30.0, 30.0), &t);
    }
}
