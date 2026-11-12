use anyhow::Result;
use clap::Parser;
use env_logger::Env;
use log::{info, warn};

use privacy_cam::pipeline::channels::{ControlMsg, PipelineChannels};
use privacy_cam::state::{AppConfig, SharedState};

/// PrivacyCam - Censura de rostros en tiempo real (Local-first, Zero-Trust)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Ejecutar sin interfaz gráfica (ideal para pruebas y servidores)
    #[arg(long, default_value_t = false)]
    list_cameras: bool,
    headless: bool,

    /// Índice de la cámara a usar (0 es usualmente la webcam por defecto)
    #[arg(short, long, default_value_t = 0)]
    camera: u32,

    /// Ruta al modelo ONNX de YOLOv8
    #[arg(short, long, default_value = "assets/models/best.onnx")]
    model_path: String,
}

fn main() -> Result<()> {
    // 1. Inicializar el sistema de logs
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("Iniciando PrivacyCam...");

    // 2. Parsear los argumentos de la línea de comandos
    let args = Cli::parse();
    // en main(), justo después de `let args = Cli::parse();`
    if args.list_cameras {
        use nokhwa::utils::ApiBackend;
        match nokhwa::query(ApiBackend::Auto) {
            Ok(cams) if !cams.is_empty() => {
                for c in cams {
                    println!("[{}] {}", c.index(), c.human_name());
                }
            }
            Ok(_) => println!("El sistema no detecta ninguna cámara."),
            Err(e) => eprintln!("No se pudo listar las cámaras: {e}"),
        }
        return Ok(());
    }
    info!(
        "Configuración CLI: Cámara: {}, Modelo: {}, Headless: {}",
        args.camera, args.model_path, args.headless
    );

    // 3. Cargar la configuración por defecto
    let config = AppConfig::default();

    // 4. Inicializar el Estado Compartido
    let shared_state = SharedState::new(config);

    // 5. Inicializar los Canales de Comunicación
    let channels = PipelineChannels::new();

    // 6. Configurar el manejador para Ctrl+C
    let ctrlc_state = shared_state.clone();
    let ctrlc_channels = channels.control_tx.clone();
    ctrlc::set_handler(move || {
        warn!("Señal de interrupción (Ctrl+C) recibida. Apagando...");
        ctrlc_state.stop();
        let _ = ctrlc_channels.send(ControlMsg::Shutdown);
    })
    .expect("Error al configurar el manejador de Ctrl+C");

    // 7. Inicializar el Detector
    info!("Cargando modelo desde: {}", args.model_path);
    let detector = privacy_cam::inference::detector::FaceDetector::new(&args.model_path)?;

    // 8. Arrancar los hilos
    // Hilo de Captura
    let capture_handle = privacy_cam::camera::capture::spawn_camera_thread(
        args.camera,
        channels.capture_tx.clone(),
        channels.control_rx.clone(),
        shared_state.clone(),
    );

    // Hilo Worker (Inferencia)
    let worker_handle = privacy_cam::pipeline::worker::spawn_worker_thread(
        detector,
        channels.capture_rx.clone(),
        channels.inference_tx.clone(),
        shared_state.clone(),
    );

    info!("Pipeline iniciado. Esperando frames procesados...");

    // 9. Iniciar la Interfaz Gráfica (GUI)
    if args.headless {
        info!("Modo Headless: No se iniciará la ventana gráfica.");
        // Si es headless, solo esperamos al Ctrl+C
        while shared_state.is_running() {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    } else {
        info!("Iniciando Interfaz Gráfica...");

        let ui_receiver = channels.inference_rx.clone();
        let native_options = eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([800.0, 600.0])
                .with_title("PrivacyCam"),
            renderer: eframe::Renderer::Glow,
            ..Default::default()
        };

        // eframe toma control del hilo principal
        let _ = eframe::run_native(
            "PrivacyCam",
            native_options,
            Box::new(move |cc| Ok(Box::new(privacy_cam::ui::PrivacyCamApp::new(cc, ui_receiver)))),
        );

        // Cuando el usuario cierra la ventana, detenemos la app
        shared_state.stop();
    }

    // 10. Esperar a que los hilos terminen limpiamente (SOLO UNA VEZ)
    let _ = capture_handle.join();
    let _ = worker_handle.join();

    info!("Aplicación cerrada correctamente.");
    Ok(())
}
