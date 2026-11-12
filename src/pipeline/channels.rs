use std::sync::Arc;
use std::time::{Duration, Instant};

use image::RgbImage;
use parking_lot::{Condvar, Mutex};

use crate::inference::detector::BoundingBox;

// ---------------------------------------------------------------------------
// Mensajes (misma API pública que antes: capture.rs, ui y main.rs no cambian)
// ---------------------------------------------------------------------------

/// Representa un fotograma de video.
/// Usamos Arc para pasar la imagen entre hilos sin copiar los píxeles.
#[derive(Clone)]
pub struct VideoFrame {
    pub image: Arc<RgbImage>,
    /// El momento exacto en que se capturó el frame. Vital para el Fail-Safe.
    pub captured_at: Instant,
}

/// Mensajes que viajan desde el Hilo de Captura hacia el Worker.
pub enum CaptureMsg {
    NewFrame(VideoFrame),
    Error(String),
}

/// Mensajes que viajan desde el Worker hacia la UI / salida de video.
pub enum InferenceMsg {
    /// El frame fue procesado (con censura aplicada) exitosamente.
    Processed(VideoFrame),
    /// Fail-safe: el detector falló o se atrasó. La salida debe congelarse.
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

// ---------------------------------------------------------------------------
// Mensajes internos entre el compositor (worker) y el hilo detector
// ---------------------------------------------------------------------------

/// Trabajo que el compositor le pasa al hilo detector.
pub struct DetectionJob {
    pub image: Arc<RgbImage>,
    pub captured_at: Instant,
}

/// Resultado del detector. Las cajas están en coordenadas de la imagen original.
pub struct DetectionResult {
    pub boxes: Vec<BoundingBox>,
    /// Momento de captura del frame sobre el que se detectó (no cuándo terminó).
    pub captured_at: Instant,
    pub inference_ms: u128,
}

// ---------------------------------------------------------------------------
// LatestSlot: buzón de un solo elemento donde "el último gana".
// ---------------------------------------------------------------------------

struct SlotState<T> {
    item: Option<T>,
    closed: bool,
}

struct SlotInner<T> {
    state: Mutex<SlotState<T>>,
    cv: Condvar,
}

/// Buzón de capacidad 1 que SOBRESCRIBE el valor anterior si nadie lo leyó.
///
/// Un canal acotado normal (`try_send`) descarta el elemento NUEVO cuando está
/// lleno, o sea que el consumidor termina trabajando con datos viejos. Para
/// video queremos lo contrario: si el detector va lento, que procese el frame
/// más reciente y se salte los intermedios.
pub struct LatestSlot<T> {
    inner: Arc<SlotInner<T>>,
}

impl<T> Clone for LatestSlot<T> {
    fn clone(&self) -> Self {
        Self { inner: Arc::clone(&self.inner) }
    }
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LatestSlot<T> {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(SlotInner {
                state: Mutex::new(SlotState { item: None, closed: false }),
                cv: Condvar::new(),
            }),
        }
    }

    /// Deja un valor, reemplazando el anterior si no había sido leído.
    pub fn put(&self, item: T) {
        {
            let mut g = self.inner.state.lock();
            g.item = Some(item);
        }
        self.inner.cv.notify_one();
    }

    /// Toma el valor si hay uno (no bloquea).
    pub fn take(&self) -> Option<T> {
        self.inner.state.lock().item.take()
    }

    /// Espera hasta `timeout` a que haya un valor. Devuelve `None` si vence el
    /// tiempo o si el buzón se cerró (usar `is_closed` para distinguir).
    pub fn wait_take(&self, timeout: Duration) -> Option<T> {
        let mut g = self.inner.state.lock();
        if g.item.is_none() && !g.closed {
            self.inner.cv.wait_for(&mut g, timeout);
        }
        g.item.take()
    }

    pub fn close(&self) {
        self.inner.state.lock().closed = true;
        self.inner.cv.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.inner.state.lock().closed
    }
}

// ---------------------------------------------------------------------------
// Canales del pipeline
// ---------------------------------------------------------------------------

/// Agrupa los extremos (Senders y Receivers) de los canales.
pub struct PipelineChannels {
    // 1. Cámara -> Worker
    pub capture_tx: crossbeam_channel::Sender<CaptureMsg>,
    pub capture_rx: crossbeam_channel::Receiver<CaptureMsg>,

    // 2. Worker -> Salida (UI / cámara virtual)
    pub inference_tx: crossbeam_channel::Sender<InferenceMsg>,
    pub inference_rx: crossbeam_channel::Receiver<InferenceMsg>,

    // 3. Control (UI -> hilos)
    pub control_tx: crossbeam_channel::Sender<ControlMsg>,
    pub control_rx: crossbeam_channel::Receiver<ControlMsg>,
}

impl PipelineChannels {
    pub fn new() -> Self {
        // Cámara -> Worker. El worker ya no espera a la IA (la detección corre
        // en otro hilo), así que consume cada frame casi al instante y esta
        // cola casi nunca se llena. Si se llena, el worker se salta al frame
        // más nuevo (ver `run_compositor`).
        let (capture_tx, capture_rx) = crossbeam_channel::bounded(2);

        // Worker -> Salida. Capacidad 1: la salida nunca acumula latencia.
        let (inference_tx, inference_rx) = crossbeam_channel::bounded(1);

        // Control: comandos pequeños y poco frecuentes.
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

impl Default for PipelineChannels {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_keeps_only_the_latest_value() {
        let slot = LatestSlot::new();
        slot.put(1);
        slot.put(2);
        slot.put(3);
        assert_eq!(slot.take(), Some(3));
        assert_eq!(slot.take(), None);
    }

    #[test]
    fn slot_wait_times_out_and_wakes_up() {
        let slot: LatestSlot<u32> = LatestSlot::new();
        assert_eq!(slot.wait_take(Duration::from_millis(10)), None);

        let s2 = slot.clone();
        let h = std::thread::spawn(move || s2.wait_take(Duration::from_secs(2)));
        std::thread::sleep(Duration::from_millis(50));
        slot.put(7);
        assert_eq!(h.join().unwrap(), Some(7));
    }

    #[test]
    fn slot_close_unblocks_waiters() {
        let slot: LatestSlot<u32> = LatestSlot::new();
        let s2 = slot.clone();
        let h = std::thread::spawn(move || s2.wait_take(Duration::from_secs(5)));
        std::thread::sleep(Duration::from_millis(50));
        slot.close();
        assert_eq!(h.join().unwrap(), None);
        assert!(slot.is_closed());
    }
}
