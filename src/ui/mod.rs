use eframe::egui;
use crossbeam_channel::Receiver;
use crate::pipeline::channels::InferenceMsg;

pub struct PrivacyCamApp {
    receiver: Receiver<InferenceMsg>,
    texture: Option<egui::TextureHandle>,
    last_frame_time: std::time::Instant,
}

impl PrivacyCamApp {
    pub fn new(cc: &eframe::CreationContext<'_>, receiver: Receiver<InferenceMsg>) -> Self {
        // Configuramos un estilo visual oscuro y limpio
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(20, 20, 20);
        cc.egui_ctx.set_visuals(visuals);

        Self {
            receiver,
            texture: None,
            last_frame_time: std::time::Instant::now(),
        }
    }
}

impl eframe::App for PrivacyCamApp {
    // CAMBIO CLAVE: Ahora el método se llama `ui` y recibe un `&mut Ui`
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // 1. Intentar recibir un nuevo frame de la IA
        if let Ok(InferenceMsg::Processed(video_frame)) = self.receiver.try_recv() {
            let size = [video_frame.image.width() as usize, video_frame.image.height() as usize];
            let pixels = video_frame.image.as_raw();

            let color_image = egui::ColorImage::from_rgb(size, pixels);

            // El contexto se obtiene a través del `ui` (ui.ctx())
            self.texture = Some(ui.ctx().load_texture(
                "video_frame",
                color_image,
                egui::TextureOptions::LINEAR,
            ));

            self.last_frame_time = std::time::Instant::now();
        }

        // 2. Dibujar la ventana usando el panel
        // CAMBIO CLAVE: CentralPanel::show ahora recibe el `ui` en lugar del `ctx`
        egui::CentralPanel::default().show(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(10.0);
                ui.heading("PrivacyCam (Local Preview)");
                ui.add_space(10.0);

                if let Some(texture) = &self.texture {
                    // Mostrar el video escalado para que quepa en la ventana
                    let available_size = ui.available_size();
                    let texture_size = texture.size_vec2();
                    let scale = (available_size.x / texture_size.x).min(available_size.y / texture_size.y);

                    ui.image(egui::load::SizedTexture::new(
                        texture.id(),
                        texture_size * scale,
                    ));
                } else {
                    ui.spinner();
                    ui.label("Esperando primer frame de la IA...");
                }

                // Mostrar el tiempo desde el último frame
                let elapsed = self.last_frame_time.elapsed().as_millis();
                if elapsed > 2000 {
                    ui.colored_label(egui::Color32::RED, format!("Advertencia: Sin frames por {} ms", elapsed));
                }
            });
        });

        // 3. Forzar a egui a redibujarse constantemente
        ui.ctx().request_repaint();
    }
}
