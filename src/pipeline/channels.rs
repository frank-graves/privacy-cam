use std::sync::Arc;
use std::time::Instant;
use image::RgbImage;

/// Representa un fotograma de video.
/// Usamos Arc (Atomic Reference Counted) para pasar la imagen entre hilos
/// sin tener que clonar los datos (evitando copias costosas en memoria).
#[derive(Clone)]
pub struct VideoFrame {
    pub image: Arc<RgbImage>,
    /// El momento exacto en que se capturó el frame. Vital para el Fail-Safe.
    pub captured_at: Instant,
}

/// Mensajes que viajan desde el Hilo de Captura hacia el Hilo de Inferencia.
pub enum CaptureMsg {
    NewFrame(VideoFrame),
    Error(String),
}

/// Mensajes que viajan desde el Hilo de Inferencia hacia el Hilo de Salida/UI.
pub enum InferenceMsg {
    /// El frame fue procesado (con o sin censura) exitosamente.
    Processed(VideoFrame),
    /// El modelo falló o tardó demasiado. El Hilo de Salida debe congelar la imagen.
    Failed(String),
}

/// Órdenes enviadas desde la UI (o CLI) hacia los hilos del pipeline.
pub enum ControlMsg {
    Pause,
    Resume,
    /// Recarga los perfiles desde el disco duro.
    ReloadProfiles,
    /// Apaga la app ordenadamente.
    Shutdown,
}

/// Estructura que agrupa los extremos (Senders y Receivers) de los canales.
/// Esto facilita pasarlos a los diferentes hilos sin perderse.
pub struct PipelineChannels {
    // 1. Cámara -> Inferencia
    pub capture_tx: crossbeam_channel::Sender<CaptureMsg>,
    pub capture_rx: crossbeam_channel::Receiver<CaptureMsg>,

    // 2. Inferencia -> Salida (OBS/UI)
    // Usamos un canal "bounded" de capacidad 1.
    // Si la inferencia produce más rápido de lo que OBS consume, descartará el frame viejo.
    pub inference_tx: crossbeam_channel::Sender<InferenceMsg>,
    pub inference_rx: crossbeam_channel::Receiver<InferenceMsg>,

    // 3. Control (UI -> Todos los hilos)
    // Usamos un canal con múltiples receptores para que la UI pueda mandar "Shutdown"
    // a la cámara, a la inferencia y a la salida al mismo tiempo.
    pub control_tx: crossbeam_channel::Sender<ControlMsg>,
    pub control_rx: crossbeam_channel::Receiver<ControlMsg>,
}

impl PipelineChannels {
    /// Inicializa todos los canales con las capacidades adecuadas.
    pub fn new() -> Self {
        // Capacidad del buffer de la cámara. Si la inferencia se atrasa,
        // no queremos guardar más de 2 frames en cola (latencia baja).
        let (capture_tx, capture_rx) = crossbeam_channel::bounded(2);

        // Capacidad 1: siempre queremos mostrar/proyectar el frame más reciente.
        let (inference_tx, inference_rx) = crossbeam_channel::bounded(1);

        // El canal de control puede ser unbounded (ilimitado) ya que los comandos
        // son pequeños y poco frecuentes (ej. click en un botón).
        let (control_tx, control_rx) = crossbeam_channel::unbounded();

        Self {
            capture_tx,
            capture_rx,
            inference_tx,
            inference_rx,
            control_tx,
            control_rx,
        }
    }
}

// Implementamos Default para poder usar PipelineChannels::default() si es necesario.
impl Default for PipelineChannels {
    fn default() -> Self {
        Self::new()
    }
}
