//! Interactive rendering: surface + egui overlay around a [`SceneRenderer`].
//!
//! Renderer is now a thin composition layer: it owns the wgpu bootstrap (adapter, device,
//! queue, surface, config), the egui renderer, and a [`SceneRenderer`] that holds the
//! actual pipelines. Every setter (`set_view`, `set_vertex_color_source`, `set_channels`) delegates to
//! the inner SceneRenderer — App's public API is unchanged. The `render` method
//! composes: SceneRenderer draws the scene pass to the swapchain, then egui draws its
//! overlay on top, then we present.
//!
//! This split exists so future non-interactive callers (PNG export, headless batch) can
//! use SceneRenderer directly against an offscreen texture, without dragging in a
//! surface or egui.

use std::sync::Arc;

use anyhow::{Context, Result};
use winit::window::Window;

use anyhow::Result as AnyResult;

use crate::density::{DensityAnchors, DensityRenderer};
use crate::exporter::{self, ExportConfig};
use crate::scene::Scene;
use crate::scene_renderer::{SceneRenderer, TextQuad};
use crate::settings::DensityParams;

/// Startup clear color, used until a theme is applied. Matches the historical dark
/// viewport backdrop so a viewer launched without any theme still looks right.
const DEFAULT_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.06,
    g: 0.06,
    b: 0.08,
    a: 1.0,
};

/// Device limits shared by both device-request sites (windowed here, headless in
/// [`crate::headless`]).
///
/// Beyond the downlevel defaults:
/// - `max_storage_buffers_per_shader_stage: 9` — Scene binds nine SSBOs in the vertex
///   stage: colors + mask + values + depth (points) plus positions + edges +
///   edge_alphas + edge_colors + edge_lengths (lines uses these to do the per-edge lookup that
///   indexed drawing can't express). Within every desktop backend's cap (Metal gives 31
///   buffer bindings per stage, Vulkan/DX12 desktop far more).
/// - Storage-binding and buffer sizes at whatever the adapter grants: positions and edges
///   are one SSBO each at 8 bytes per element, so the 128 MiB default caps rendering at
///   ~16.7M vertices/edges. Real graphs exceed that (road: 18M vertices / 22.2M edges).
/// - `using_resolution` raises `max_texture_dimension_{1d,2d,3d}` so big exports work.
pub fn request_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let adapter_limits = adapter.limits();
    wgpu::Limits {
        max_storage_buffers_per_shader_stage: 9,
        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        max_buffer_size: adapter_limits.max_buffer_size,
        ..wgpu::Limits::downlevel_defaults()
    }
    .using_resolution(adapter_limits)
}

/// One-line startup report of the size limits the device actually granted — the numbers
/// [`crate::scene::check_device_limits`] tests against, so a "graph too large" error can
/// be cross-checked against what this printed.
pub fn print_granted_limits(device: &wgpu::Device) {
    let l = device.limits();
    println!(
        "wgpu: granted max storage binding {} MiB, max buffer {} MiB",
        l.max_storage_buffer_binding_size as u64 / (1024 * 1024),
        l.max_buffer_size / (1024 * 1024),
    );
}

pub struct Renderer {
    scene_renderer: SceneRenderer,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,

    /// Adapter-supported surface dimension cap. `Surface::configure` must never receive a
    /// width or height above this — validation error otherwise. Cached at construction so
    /// `resize` can clamp without another adapter query.
    max_surface_dim: u32,

    /// Viewport background. Written by [`Renderer::set_clear_color`] whenever the theme
    /// changes; read on every render frame.
    clear_color: wgpu::Color,

    /// Lazily-built density strategy — `None` until the first density frame. Kept here (not
    /// in App) so App's per-frame borrows stay simple: App holds only the density PARAMS +
    /// cached anchors, and hands them in.
    density_renderer: Option<DensityRenderer>,

    egui_renderer: egui_wgpu::Renderer,
}

/// Per-frame UI data handed to `Renderer::render`. Held by-reference so App can pass a
/// tessellated frame without moving ownership of the paint jobs into the renderer.
pub struct UiFrame<'a> {
    pub paint_jobs: &'a [egui::epaint::ClippedPrimitive],
    pub textures_delta: &'a egui::TexturesDelta,
    pub pixels_per_point: f32,
}

impl Renderer {
    /// Bring up wgpu: adapter, device, queue, surface. Hand device + queue + surface
    /// format to [`SceneRenderer`], which builds the pipelines. Independent of scene
    /// data — Scene is created separately once the first topology publish arrives.
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no suitable GPU adapter")?;

        let limits = request_limits(&adapter);

        let max_surface_dim = limits.max_texture_dimension_2d;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wembed-viewer device"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;
        print_granted_limits(&device);

        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.clamp(1, max_surface_dim),
            height: size.height.clamp(1, max_surface_dim),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // egui uses a separate pipeline internally; we just hand it device + format at
        // setup. `dithering = true` matches egui's default; visually indistinguishable
        // for our overlay but avoids a warning.
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, true);

        // SceneRenderer clones device + queue (both Arc-backed in wgpu) so it can push
        // uniform updates independently of Renderer.
        let scene_renderer = SceneRenderer::new(device, queue, format);

        Ok(Self {
            scene_renderer,
            surface,
            config,
            max_surface_dim,
            clear_color: DEFAULT_CLEAR_COLOR,
            density_renderer: None,
            egui_renderer,
        })
    }

    /// Update the viewport clear color — driven by the current theme's `base00`. Applied
    /// on the next render frame.
    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Resize the surface. Does NOT update view uniforms — the App is responsible for
    /// pushing a fresh `set_view` after resize since view_proj depends on window aspect.
    /// Width and height are clamped to `max_surface_dim` (adapter's texture limit) so
    /// resizing to fullscreen on a display wider than the GPU can back doesn't blow up
    /// `Surface::configure`.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.clamp(1, self.max_surface_dim);
        self.config.height = height.clamp(1, self.max_surface_dim);
        self.surface
            .configure(self.scene_renderer.device(), &self.config);
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    // ---- delegated setters + accessors --------------------------------------

    pub fn device(&self) -> &wgpu::Device {
        self.scene_renderer.device()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.scene_renderer.queue()
    }

    pub fn scene_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        self.scene_renderer.scene_bind_group_layout()
    }

    pub fn set_view(&mut self, view: &crate::framing::View) {
        self.scene_renderer.set_view(view);
    }

    pub fn set_vertex_color_source(&mut self, source: u32) {
        self.scene_renderer.set_vertex_color_source(source);
    }

    pub fn set_channels(&mut self, channels: &crate::scene_renderer::Channels) {
        self.scene_renderer.set_channels(channels);
    }

    /// Update the `theme` edge-color source's RGB — delegate to the inner [`SceneRenderer`].
    pub fn set_depth_fade(&mut self, depth_fade: f32) {
        self.scene_renderer.set_depth_fade(depth_fade);
    }

    pub fn set_theme_colors(&mut self, theme: &crate::theme::Theme) {
        self.scene_renderer.set_theme_colors(theme);
    }

    /// Upload a rasterized title bitmap — delegate to the inner [`SceneRenderer`].
    pub fn make_text_quad(&self, bitmap: &crate::text::TextBitmap) -> TextQuad {
        self.scene_renderer.make_text_quad(bitmap)
    }

    /// Render the scene to a PNG at `config.path`. Delegates to [`crate::exporter`] with
    /// our inner [`SceneRenderer`]. Mutates the uniform buffer (view + point_radius) —
    /// the caller's next frame overwrites this, so no restore is needed. `texts` are the
    /// screen-anchored quads (title, legend) rasterized at export scale plus their NDC
    /// rects in the export target.
    pub fn export_png(
        &mut self,
        scene: &Scene,
        view: &crate::framing::View,
        config: &ExportConfig,
        texts: &[(&TextQuad, [f32; 4])],
    ) -> AnyResult<()> {
        exporter::export_png(&mut self.scene_renderer, scene, view, config, texts)
    }

    /// Lazily build the internal [`DensityRenderer`] against this renderer's layouts +
    /// surface format. Idempotent; cloning device/queue is cheap (Arc-backed).
    fn ensure_density_renderer(&mut self) {
        if self.density_renderer.is_none() {
            self.density_renderer = Some(DensityRenderer::new(
                self.scene_renderer.device().clone(),
                self.scene_renderer.queue().clone(),
                self.scene_renderer.uniform_bind_group_layout(),
                self.scene_renderer.scene_bind_group_layout(),
                self.config.format,
            ));
        }
    }

    /// Debounced auto-exposure: accumulate + blur + read back the density, compute
    /// percentile anchors from the histogram (over lit pixels). Called by App only when the
    /// view has been at rest — a readback stalls the pipeline. Returns `None` if the read
    /// fails (device busy); App keeps its previous anchors then.
    pub fn recompute_density_anchors(
        &mut self,
        scene: &Scene,
        params: &DensityParams,
    ) -> Option<DensityAnchors> {
        self.ensure_density_renderer();
        let (w, h) = (self.config.width, self.config.height);
        let device = self.scene_renderer.device().clone();
        let queue = self.scene_renderer.queue().clone();
        let density = self.density_renderer.as_mut().expect("just ensured");
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("density exposure encoder"),
        });
        let readback = density.accumulate_and_readback(
            &mut encoder,
            scene,
            self.scene_renderer.uniform_bind_group(),
            w,
            h,
            params.sigma,
        );
        let idx = queue.submit(std::iter::once(encoder.finish()));
        let field = readback.read(&device, idx).ok()?;
        Some(crate::density::compute_anchors(params, &field))
    }

    // ---- per-frame render ---------------------------------------------------

    /// Density-mode frame: accumulate + blur + tonemap to the swapchain (using cached
    /// `anchors`), text on top, then egui. Mirrors [`Renderer::render`] but swaps the scene
    /// pass for the density pipeline. `ramp` comes from the active theme/config.
    #[allow(clippy::too_many_arguments)]
    pub fn render_density(
        &mut self,
        scene: &Scene,
        ui: &UiFrame<'_>,
        texts: &[(&TextQuad, [f32; 4])],
        params: &DensityParams,
        anchors: DensityAnchors,
        ramp: &[[u8; 4]],
    ) -> Result<(), wgpu::SurfaceError> {
        self.ensure_density_renderer();
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let device = self.scene_renderer.device();
        let queue = self.scene_renderer.queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("density frame"),
        });

        for (id, delta) in &ui.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, delta);
        }
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui.pixels_per_point,
        };
        self.egui_renderer
            .update_buffers(device, queue, &mut encoder, ui.paint_jobs, &screen_desc);

        // density pass (tonemap clears + writes the whole target), then text on top.
        let (w, h) = (self.config.width, self.config.height);
        let density = self.density_renderer.as_mut().expect("ensured above");
        density.render_frame(
            &mut encoder,
            &view,
            scene,
            self.scene_renderer.uniform_bind_group(),
            w,
            h,
            params.sigma,
            anchors,
            ramp,
        );
        self.scene_renderer
            .draw_texts_load(&mut encoder, &view, texts);

        // egui overlay.
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut pass, ui.paint_jobs, &screen_desc);
        }
        for id in &ui.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    pub fn render(
        &mut self,
        scene: &Scene,
        ui: &UiFrame<'_>,
        texts: &[(&TextQuad, [f32; 4])],
    ) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let device = self.scene_renderer.device();
        let queue = self.scene_renderer.queue();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // ---- egui texture + buffer prep (must precede the render passes) ----
        for (id, delta) in &ui.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, delta);
        }
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: ui.pixels_per_point,
        };
        self.egui_renderer
            .update_buffers(device, queue, &mut encoder, ui.paint_jobs, &screen_desc);

        // ---- scene pass (delegated to SceneRenderer) ----
        // No MSAA on the surface path — the swapchain is single-sample, and vertex disks
        // are already antialiased in the fragment shader. Offscreen export uses MSAA.
        self.scene_renderer
            .render(&mut encoder, &view, None, scene, self.clear_color, texts);

        // ---- egui pass ----
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut pass, ui.paint_jobs, &screen_desc);
        }

        // ---- cleanup textures egui asked us to free ----
        for id in &ui.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }
}
