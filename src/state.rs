use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Define a quién debemos censurar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CensorMode {
    /// Censura todas las caras detectadas (Modo Privacidad Total).
    All,
    /// Censura solo a las personas que estén en la lista de perfiles.
    Blacklist,
    /// Censura a todos EXCEPTO a las personas en la lista de perfiles.
    Whitelist,
}

/// Configuraciones ajustables por el usuario (se pueden guardar en un archivo TOML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub censor_mode: CensorMode,
    /// Tiempo máximo (en milisegundos) que puede tardar la IA antes de activar el Fail-Safe.
    pub fail_safe_timeout_ms: u64,
    /// Fuerza del desenfoque (sigma del filtro gaussiano).
    pub blur_strength: f32,
    /// Si es true, la app requiere intervención manual para reanudar después de un Fallo.
    pub require_manual_resume: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            censor_mode: CensorMode::All,
            fail_safe_timeout_ms: 200, // 200ms es un buen límite para video en tiempo real
            blur_strength: 15.0,
            require_manual_resume: false,
        }
    }
}

/// Los estados operativos de la aplicación.
#[derive(Debug, Clone, PartialEq)]
pub enum AppState {
    /// Todo funciona correctamente. La IA procesa y censura frames.
    Normal,
    /// El usuario pausó la app manualmente.
    Paused,
    /// FAIL-SAFE: La IA falló, tardó demasiado o perdió un rostro.
    /// El video se congela hasta que se resuelva.
    Freeze { reason: String },
    /// Apagando la aplicación de forma segura.
    Shutdown,
}

/// El estado interno que será protegido por el RwLock.
struct InternalState {
    pub current_state: AppState,
    pub config: AppConfig,
}

/// El contenedor seguro (Thread-Safe) que compartiremos entre hilos.
/// Es barato de clonar (solo incrementa un contador atómico).
#[derive(Clone)]
pub struct SharedState {
    inner: Arc<RwLock<InternalState>>,
}

impl SharedState {
    /// Inicializa el estado con la configuración por defecto.
    pub fn new(config: AppConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InternalState {
                current_state: AppState::Normal,
                config,
            })),
        }
    }

    /// Obtiene el estado actual.
    pub fn get_state(&self) -> AppState {
        self.inner.read().current_state.clone()
    }

    /// Cambia el estado de la aplicación (ej. de Normal a Freeze).
    pub fn set_state(&self, new_state: AppState) {
        let mut inner = self.inner.write();
        // Si el estado cambia, lo registramos (útil para la UI y logs)
        if inner.current_state != new_state {
            inner.current_state = new_state;
        }
    }

    /// Obtiene una copia de la configuración actual.
    pub fn get_config(&self) -> AppConfig {
        self.inner.read().config.clone()
    }

    /// Actualiza la configuración en tiempo real desde la UI.
    pub fn update_config(&self, new_config: AppConfig) {
        let mut inner = self.inner.write();
        inner.config = new_config;
    }
}
