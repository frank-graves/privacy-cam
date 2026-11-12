use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use log::{error, info, warn};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, CameraFormat, FrameFormat, RequestedFormat, RequestedFormatType, Resolution},
    Camera,
};

use crate::pipeline::channels::{CaptureMsg, ControlMsg, VideoFrame};
use crate::state::SharedState;

pub fn spawn_camera_thread(
    camera_index: u32,
    tx: Sender<CaptureMsg>,
    control_rx: Receiver<ControlMsg>,
    state: SharedState,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Err(e) = run_camera_loop(camera_index, tx, control_rx, state) {
            error!("El hilo de captura falló críticamente: {}", e);
        }
    })
}

// AQUÍ ESTABA EL ERROR: Faltaban los parámetros de la función
fn run_camera_loop(
    camera_index: u32,
    tx: Sender<CaptureMsg>,
    control_rx: Receiver<ControlMsg>,
    state: SharedState,
) -> Result<()> {
    info!("Inicializando webcam en el índice {}", camera_index);

    let camera_index_enum = CameraIndex::Index(camera_index);

    // Intento 1: 640x480 MJPEG
    let req_mjpeg = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::Closest(CameraFormat::new(Resolution::new(640, 480), FrameFormat::MJPEG, 30))
    );

    // Intento 2: 640x480 YUYV
    let req_yuyv = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::Closest(CameraFormat::new(Resolution::new(640, 480), FrameFormat::YUYV, 30))
    );

    // Intento 3: 1280x720 MJPEG
    let req_720p = RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::Closest(CameraFormat::new(Resolution::new(1280, 720), FrameFormat::MJPEG, 30))
    );

    // Intento 4: Formato por defecto
    let req_default = RequestedFormat::new::<RgbFormat>(RequestedFormatType::None);

    let mut camera = match Camera::new(camera_index_enum.clone(), req_mjpeg) {
        Ok(cam) => { info!("Cámara abierta usando MJPEG 640x480."); cam }
        Err(_) => match Camera::new(camera_index_enum.clone(), req_yuyv) {
            Ok(cam) => { info!("Cámara abierta usando YUYV 640x480."); cam }
            Err(_) => match Camera::new(camera_index_enum.clone(), req_720p) {
                Ok(cam) => { info!("Cámara abierta usando MJPEG 1280x720."); cam }
                Err(_) => {
                    warn!("Cámara abierta en su formato por defecto (posiblemente 1080p). ¡Advertencia de rendimiento!");
                    Camera::new(camera_index_enum, req_default)?
                }
            }
        }
    };

    camera.open_stream()?;
    info!("Stream de cámara iniciado. Resolución real: {:?}", camera.resolution());

    loop {
        if !state.is_running() {
            info!("Hilo de captura recibiendo orden de apagado.");
            break;
        }

        if let Ok(ControlMsg::Shutdown) = control_rx.try_recv() {
            info!("Hilo de captura recibiendo orden de apagado (canal).");
            break;
        }

        match camera.frame() {
            Ok(buffer) => {
                match buffer.decode_image::<RgbFormat>() {
                    Ok(img) => {
                        let frame = VideoFrame {
                            image: Arc::new(img),
                            captured_at: Instant::now(),
                        };

                        if tx.try_send(CaptureMsg::NewFrame(frame)).is_err() {
                            // Canal lleno
                        }
                    }
                    Err(e) => {
                        error!("Error decodificando imagen: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Error leyendo frame: {}", e);

                let err_msg = e.to_string();
                if err_msg.contains("0xC00D3EA2") || err_msg.contains("no longer present") {
                    error!("La cámara se ha desconectado. Deteniendo hilo de captura.");
                    let _ = tx.send(CaptureMsg::Error("Cámara desconectada".to_string()));
                    break;
                }

                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    Ok(())
}
