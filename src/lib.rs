//! # PrivacyCam Core
//!
//! Una librería para la censura selectiva de rostros en tiempo real.
//! Diseñada bajo la filosofía Local-first y Zero-Trust.

// Exponemos los módulos públicos para que puedan ser usados por main.rs y los tests.
pub mod camera;
pub mod inference;
pub mod pipeline;
pub mod profiles;
pub mod state;
pub mod ui;
