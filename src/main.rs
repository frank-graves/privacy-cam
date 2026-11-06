use anyhow::{Context, Result};
use clap::Parser;
use env_logger::Env;
use log::{error, info, warn};

// Importamos nuestros propios módulos (esto requerirá que configuremos lib.rs luego)
use privacy_cam::pipeline::channels::{ControlMsg, PipelineChannels};
use privacy_cam::state::{AppConfig, SharedState};

/// PrivacyCam - Censura de rostros en tiempo real (Local-first, Zero-Trust)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Ejecutar sin interfaz gráfica (ideal para pruebas y servidores)
    #[arg(long, default_value_t = false)]
    headless: bool,

    /// Índice de la cámara a usar (0 es usualmente la webcam por defecto)
    #[arg(short, long, default_value_t = 0)]
    camera: u32,

    /// Ruta al modelo ONNX de YOLOv8
    #[arg(short, long, default_value = "assets/models/best.onnx")]
    model_path: String,
}

fn main() -> Result<()> {
    // 1. Inicializar el sistema de logs (útil para ver qué pasa en los hilos)
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    info!("Iniciando PrivacyCam...");

    // 2. Parsear los argumentos de la línea de comandos
    let args = Cli::parse();
    info!("Configuración CLI: Cámara: {}, Modelo: {}, Headless: {}", args.camera, args.model_path, args.headless);

    // 3. Cargar la configuración por defecto (en el futuro, leeremos de config/settings.toml)
    let config = AppConfig::default();

    // 4. Inicializar el Estado Compartido (El cerebro seguro)
    let shared_state = SharedState::new(config);

    // 5. Inicializar los Canales de Comunicación (Las tuberías)
    let channels = PipelineChannels::new();

    // 6. Configurar el manejador para Ctrl+C (Apagado seguro)
    // Esto asegura que no dejemos la cámara virtual colgada si cerramos abruptamente.
    let ctrlc_channels = channels.control_tx.clone();
    ctrlc::set_handler(move || {
        warn!("Señal de interrupción (Ctrl+C) recibida. Apagando...");
        let _ = ctrlc_channels.send(ControlMsg::Shutdown);
    }).expect("Error al configurar el manejador de Ctrl+C");

    // =====================================================================
    // FASE 2 y 3: Aquí arrancaremos los hilos reales más adelante.
    // =====================================================================
    info!("Sistema inicializado correctamente. Esperando hilos...");

    // Simulamos que los hilos están corriendo escuchando el canal de control.
    // En las siguientes fases, aquí haremos thread::spawn() para:
    // 1. hilo_captura(channels.capture_tx, ...)
    // 2. hilo_inferencia(channels.capture_rx, channels.inference_tx, shared_state.clone(), ...)
    // 3. hilo_salida(channels.inference_rx, shared_state.clone(), ...)

    if !args.headless {
        info!("El modo GUI está desactivado en esta fase. Usa --headless para evitar este mensaje.");
    }

    // 7. Bucle principal (espera al comando de Shutdown)
    // Aquí la UI (o la terminal) se quedaría esperando eventos.
    loop {
        crossbeam_channel::select! {
            recv(channels.control_rx) -> msg => {
                match msg {
                    Ok(ControlMsg::Shutdown) => {
                        info!("Apagado ordenado iniciado...");
                        shared_state.set_state(privacy_cam::state::AppState::Shutdown);
                        break;
                    },
                    Ok(_) => {
                        // Aquí manejaremos Pause, Resume, etc.
                    },
                    Err(_) => {
                        error!("Canal de control desconectado inesperadamente.");
                        break;
                    }
                }
            }
            // Aquí podríamos escuchar otros canales si la UI es headless
            default(std::time::Duration::from_millis(100)) => {
                // Tick del bucle principal
            }
        }
    }

    info!("PrivacyCam finalizado con éxito. Memoria liberada.");
    Ok(())
}
