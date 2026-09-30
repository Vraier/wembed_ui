//! Surface-independent scene rendering: pipelines, uniforms, and the scene pass.
//!
//! `SceneRenderer` owns everything the scene draw needs — points and edges pipelines,
//! the uniform buffer, the scene bind-group layout, and clones of the wgpu Device and
//! Queue. It does NOT own a surface, an egui renderer, or any window-tied state. This
//! split exists so the same rendering machinery can drive:
//!   - the interactive viewer (wrapped by [`crate::render::Renderer`] which adds surface
//!     + egui + present),
//!   - the future PNG export path (renders to an offscreen texture at an arbitrary
//!     resolution),
//!   - and a future headless / batch mode (no window, no egui, just render-and-save).
//!
//! Device and Queue are stored by value here. Both are Arc-backed in wgpu 24+, so cloning
//! them is cheap and gives us two owners (Renderer keeps the primary; SceneRenderer keeps
//! its own for `queue.write_buffer` uniform updates without needing `&Queue` on every
//! setter).
//!
//! The pipeline layouts are built against a caller-provided target color format so both
//! the sRGB surface path and an `Rgba8UnormSrgb` offscreen path can share the same pipes
//! — as long as the caller uses the same format for both.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::scene::{self, Scene};

/// MSAA sample count used by the offscreen (export) render path. `4` is sufficient
/// for our vertex disks (which are already antialiased in the fragment shader via
/// smoothstep) and for line edges at typical wallpaper resolutions. Bumping to `8`
/// doubles the offscreen memory cost and starts hitting adapter limits on
/// non-workstation GPUs; not worth it until we see visible aliasing.
pub const MSAA_SAMPLE_COUNT: u32 = 4;

/// Shader-facing uniforms bound as `@group(0) @binding(0)`. See `shaders/points.wgsl` and
/// `shaders/lines.wgsl` for the mirroring WGSL structs — layout must stay in lockstep (160
/// bytes, multiple of 16).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    point_radius: f32,
    /// `0` = theme vertex color, `1` = per-vertex colors SSBO. Mirrors `VERTEX_COLOR_*` in
    /// `points.wgsl`.
    vertex_color_source: u32,
    /// Size channel selector — `0` = constant, `1` = per-vertex `sqrt(w/mean_w)`.
    vertex_size_source: u32,
    /// Edge-alpha channel selector — `0` = constant, `1` = precomputed `1/(w_a*w_b)`.
    edge_alpha_source: u32,
    /// Global multiplier on point radius (UI slider). Composes with size-by-weight.
    vertex_size_scale: f32,
    /// Global multiplier on point alpha (UI slider). `0.0` = vertices invisible.
    vertex_alpha_scale: f32,
    /// Global multiplier on edge base alpha (UI slider). `0.0` = edges invisible.
    edge_alpha_scale: f32,
    /// Edge-color source selector. Mirrors `EDGE_COLOR_*` in `lines.wgsl` /
    /// `cli::EdgeColorSource::shader_value` — `0` theme, `1` vertex, `2` file, `3` length.
    edge_color_source: u32,
    /// Theme edge color for the `theme` source (rgb; a unused). Normalized [0,1] bytes.
    edge_theme_color: [f32; 4],
    /// `ln(r)` clamp bounds for the `length` source: `log_min = ln(range.lo)` (< 0),
    /// `log_max = ln(range.hi)` (> 0). `r = 1` is the diverging colormap's neutral midpoint.
    edge_len_log_min: f32,
    edge_len_log_max: f32,
    /// Depth-shading strength (0 = off): vertex alpha × `1 - depth_fade * depth[i]`,
    /// fading vertices the 2D projection misrepresents. Only nonzero at dimension > 2.
    depth_fade: f32,
    /// Edge stroke width in device pixels. `1.0` at `framing::REFERENCE_HEIGHT` reproduces
    /// the old 1-device-pixel hardware line; scales with render height. Points ignore it.
    line_width_px: f32,
    /// Render-target size in device pixels (width, height) — the lines shader converts clip
    /// endpoints to pixel space to build the perpendicular offset. Points ignore it.
    viewport_size: [f32; 2],
    /// Global multiplier on `line_width_px` (UI slider).
    edge_width_scale: f32,
    _pad1: f32,
    /// Vertex color of the `theme` vertex color source (rgb + alpha), normalized sRGB bytes.
    vertex_theme_color: [f32; 4],
}

/// Channel selectors and multipliers pushed by [`SceneRenderer::set_channels`], all derived
/// from the visual state (`VisualState::channels`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Channels {
    pub vertex_size_source: u32,
    pub edge_alpha_source: u32,
    pub edge_color_source: u32,
    pub vertex_size_scale: f32,
    pub vertex_alpha_scale: f32,
    pub edge_alpha_scale: f32,
    pub edge_width_scale: f32,
    pub edge_len_log_min: f32,
    pub edge_len_log_max: f32,
}

pub struct SceneRenderer {
    // Cloned from Renderer at construction. Arc-backed in wgpu; cheap to hold a second
    // owner so setters don't need `&Queue` on every call.
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Two pipeline sets: `_1x` renders directly to a single-sample target (the swapchain
    // surface), `_msaa` renders to a multi-sample target with a resolve step (the export
    // path). Pipelines are otherwise identical — only `multisample.count` differs, and
    // wgpu enforces that count matches the render pass's attachment sample count.
    points_pipeline_1x: wgpu::RenderPipeline,
    edges_pipeline_1x: wgpu::RenderPipeline,
    points_pipeline_msaa: wgpu::RenderPipeline,
    edges_pipeline_msaa: wgpu::RenderPipeline,
    text_pipeline_1x: wgpu::RenderPipeline,
    text_pipeline_msaa: wgpu::RenderPipeline,

    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    scene_bind_group_layout: wgpu::BindGroupLayout,

    // Text-quad shared state: bind-group layout + sampler. The rect uniform and texture
    // live on each [`TextQuad`] (per-quad rect buffers let one pass draw several quads —
    // title + legend — each with its own placement).
    text_bind_group_layout: wgpu::BindGroupLayout,
    text_sampler: wgpu::Sampler,

    /// Target color format used to build the pipelines. Exposed so the exporter can
    /// allocate its offscreen texture with a matching format.
    target_format: wgpu::TextureFormat,

    current_uniforms: Uniforms,
}

/// A CPU-rasterized text bitmap uploaded as a texture, ready for the text quad pass.
/// Built via [`SceneRenderer::make_text_quad`]. Owns its NDC-rect uniform buffer —
/// rewritten by every render call that draws it, so the same quad can land at different
/// rects in the window and export passes, and several quads (title + legend) can draw in
/// one pass without clobbering each other's placement.
pub struct TextQuad {
    // Held alive for the bind group's sake; never read after creation.
    _texture: wgpu::Texture,
    rect_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SceneRenderer {
    /// Build the pipeline set against `target_format`. Renderer passes its swapchain
    /// format here; a future offscreen exporter would pass the offscreen texture's format
    /// (both must match if the pipelines are to be shared — otherwise a second
    /// SceneRenderer with a different format is needed).
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let current_uniforms = Uniforms {
            view_proj: [[0.0; 4]; 4],
            point_radius: 0.0,
            vertex_color_source: 0,
            vertex_size_source: 0,
            edge_alpha_source: 0,
            vertex_size_scale: 1.0,
            vertex_alpha_scale: 1.0,
            edge_alpha_scale: 0.35,
            edge_color_source: 0,
            // Old hardcoded edge floats; overwritten by the first theme push in App/headless.
            edge_theme_color: [0.40, 0.45, 0.55, 1.0],
            // ln(0.25) / ln(4) — the default r ∈ [1/4, 4] stress window.
            edge_len_log_min: -1.386_294_4,
            edge_len_log_max: 1.386_294_4,
            depth_fade: 0.0,
            // Overwritten by the first `set_view` (target dims known there). 1px / 1080p ref
            // matches the old hardware-line look until a real target size arrives.
            line_width_px: 1.0,
            viewport_size: [1920.0, 1080.0],
            edge_width_scale: 1.0,
            _pad1: 0.0,
            vertex_theme_color: [0.95, 0.55, 0.15, 1.0],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniforms"),
            contents: bytemuck::bytes_of(&current_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let scene_bind_group_layout =
            device.create_bind_group_layout(&scene::bind_group_layout_desc());

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shared pipeline layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &scene_bind_group_layout],
            push_constant_ranges: &[],
        });

        let points_pipeline_1x = build_points_pipeline(&device, &pipeline_layout, target_format, 1);
        let edges_pipeline_1x = build_edges_pipeline(&device, &pipeline_layout, target_format, 1);
        let points_pipeline_msaa =
            build_points_pipeline(&device, &pipeline_layout, target_format, MSAA_SAMPLE_COUNT);
        let edges_pipeline_msaa =
            build_edges_pipeline(&device, &pipeline_layout, target_format, MSAA_SAMPLE_COUNT);

        // Text-quad state. The sampler is Nearest because bitmaps are composited 1:1
        // (exports re-rasterize at target scale instead of stretching).
        let text_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("text sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let text_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("text bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let text_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text pipeline layout"),
            bind_group_layouts: &[&text_bind_group_layout],
            push_constant_ranges: &[],
        });
        let text_pipeline_1x =
            build_text_pipeline(&device, &text_pipeline_layout, target_format, 1);
        let text_pipeline_msaa = build_text_pipeline(
            &device,
            &text_pipeline_layout,
            target_format,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            device,
            queue,
            points_pipeline_1x,
            edges_pipeline_1x,
            points_pipeline_msaa,
            edges_pipeline_msaa,
            text_pipeline_1x,
            text_pipeline_msaa,
            uniform_buffer,
            uniform_bind_group,
            uniform_bind_group_layout,
            scene_bind_group_layout,
            text_bind_group_layout,
            text_sampler,
            target_format,
            current_uniforms,
        }
    }

    /// Upload a rasterized text bitmap as a texture + bind group, ready to hand to
    /// [`SceneRenderer::render`]. Rebuild whenever the bitmap changes (title edit, theme
    /// color change, scale change); dropping the old quad frees its texture.
    pub fn make_text_quad(&self, bitmap: &crate::text::TextBitmap) -> TextQuad {
        let size = wgpu::Extent3d {
            width: bitmap.width,
            height: bitmap.height,
            depth_or_array_layers: 1,
        };
        // sRGB view format: the bitmap holds sRGB color bytes; sampling converts to
        // linear so blending on the sRGB target matches everywhere.
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("title text"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bitmap.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bitmap.width * 4),
                rows_per_image: Some(bitmap.height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // Per-quad rect uniform (16 bytes) so several quads can draw in one pass.
        let rect_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text rect"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("text bind group"),
            layout: &self.text_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rect_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.text_sampler),
                },
            ],
        });
        TextQuad {
            _texture: texture,
            rect_buffer,
            bind_group,
        }
    }

    /// Color format the pipelines were built against — the offscreen exporter allocates
    /// its target texture with this format so pipeline compatibility holds.
    pub fn target_format(&self) -> wgpu::TextureFormat {
        self.target_format
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn scene_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scene_bind_group_layout
    }

    /// The `@group(0)` uniform bind-group layout — the density strategy builds its
    /// accumulation pipelines against it so they read the same view/channel uniforms.
    pub fn uniform_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.uniform_bind_group_layout
    }

    /// The `@group(0)` uniform bind group — density accumulation binds it verbatim, so the
    /// splat geometry tracks the normal pipelines' view/channel state exactly.
    pub fn uniform_bind_group(&self) -> &wgpu::BindGroup {
        &self.uniform_bind_group
    }

    /// Update view_proj + point radius + edge-quad viewport/width uniforms. Call whenever the
    /// camera state or render-target size changes.
    pub fn set_view(&mut self, view: &crate::framing::View) {
        self.current_uniforms.view_proj = view.view_proj.to_cols_array_2d();
        self.current_uniforms.point_radius = view.point_radius;
        self.current_uniforms.viewport_size = [view.width as f32, view.height as f32];
        self.current_uniforms.line_width_px = view.line_width_px;
        self.upload_uniforms();
    }

    /// Value comes from `cli::VertexColorSource::shader_value()` of the *effective* source.
    pub fn set_vertex_color_source(&mut self, source: u32) {
        if self.current_uniforms.vertex_color_source == source {
            return;
        }
        self.current_uniforms.vertex_color_source = source;
        self.upload_uniforms();
    }

    /// Update the size + alpha + width + edge-color channels — the source selectors (which
    /// shader branch runs), their scale multipliers (UI sliders), and the `length`-source
    /// `ln(r)` clamp window. The theme colors are pushed separately via
    /// [`SceneRenderer::set_theme_colors`].
    pub fn set_channels(&mut self, c: &Channels) {
        let u = &mut self.current_uniforms;
        let unchanged = u.vertex_size_source == c.vertex_size_source
            && u.edge_alpha_source == c.edge_alpha_source
            && u.edge_color_source == c.edge_color_source
            && u.vertex_size_scale == c.vertex_size_scale
            && u.vertex_alpha_scale == c.vertex_alpha_scale
            && u.edge_alpha_scale == c.edge_alpha_scale
            && u.edge_width_scale == c.edge_width_scale
            && u.edge_len_log_min == c.edge_len_log_min
            && u.edge_len_log_max == c.edge_len_log_max;
        if unchanged {
            return;
        }
        u.vertex_size_source = c.vertex_size_source;
        u.edge_alpha_source = c.edge_alpha_source;
        u.edge_color_source = c.edge_color_source;
        u.vertex_size_scale = c.vertex_size_scale;
        u.vertex_alpha_scale = c.vertex_alpha_scale;
        u.edge_alpha_scale = c.edge_alpha_scale;
        u.edge_width_scale = c.edge_width_scale;
        u.edge_len_log_min = c.edge_len_log_min;
        u.edge_len_log_max = c.edge_len_log_max;
        self.upload_uniforms();
    }

    /// Update the depth-shading strength (0 = off). Pushed by App/headless whenever the
    /// visual state changes; stays 0 at dimension 2 so the shader multiply is a no-op.
    pub fn set_depth_fade(&mut self, depth_fade: f32) {
        if self.current_uniforms.depth_fade == depth_fade {
            return;
        }
        self.current_uniforms.depth_fade = depth_fade;
        self.upload_uniforms();
    }

    /// Push the theme's default vertex color and edge color. Called on the initial theme
    /// apply and on every runtime theme switch, independent of the
    /// [`SceneRenderer::set_channels`] visual state.
    pub fn set_theme_colors(&mut self, theme: &crate::theme::Theme) {
        let vertex = crate::theme::rgba_to_unorm(theme.vertex_color());
        let edge = crate::theme::rgba_to_unorm(theme.edge_color());
        if self.current_uniforms.vertex_theme_color == vertex
            && self.current_uniforms.edge_theme_color == edge
        {
            return;
        }
        self.current_uniforms.vertex_theme_color = vertex;
        self.current_uniforms.edge_theme_color = edge;
        self.upload_uniforms();
    }

    fn upload_uniforms(&self) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.current_uniforms),
        );
    }

    /// Encode the scene pass (edges, then points, then the optional title quad) into
    /// `encoder`, clearing `target_view` to `clear_color` first. Doesn't submit — the
    /// caller composes further passes (egui in interactive mode, a colormap post-pass in
    /// the future density strategy) and submits the whole encoder at once.
    /// If `resolve_view` is `Some`, `target_view` is a multisample texture and the
    /// hardware resolves to `resolve_view` at end-of-pass — used for the offscreen
    /// export path. If `None`, `target_view` is single-sample — used for the swapchain
    /// surface path. Pipeline sample-count matches the render pass sample-count in either
    /// case; wgpu validates this at pass creation.
    ///
    /// `texts` are the screen-anchored quads (title, legend, …), each with its NDC rect
    /// for THIS target. Every quad's own rect uniform is rewritten here on every call —
    /// window and export passes use different rects, and each render is its own
    /// submission, so the writes always land before their pass.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        resolve_view: Option<&wgpu::TextureView>,
        scene: &Scene,
        clear_color: wgpu::Color,
        texts: &[(&TextQuad, [f32; 4])],
    ) {
        for (quad, rect) in texts {
            self.queue
                .write_buffer(&quad.rect_buffer, 0, bytemuck::cast_slice(rect));
        }
        let (points_pipeline, edges_pipeline) = if resolve_view.is_some() {
            (&self.points_pipeline_msaa, &self.edges_pipeline_msaa)
        } else {
            (&self.points_pipeline_1x, &self.edges_pipeline_1x)
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: resolve_view,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Edges first so points draw on top. Visibility is a shader-side alpha
        // multiplier (0.0 = invisible via blending) — no pipeline gate here.
        //
        // Non-indexed instanced draw: `num_edges` instances × 4 vertices per instance —
        // each edge is a screen-space quad (triangle strip) so the stroke width can scale
        // with render height (see lines.wgsl header). All per-edge data (endpoints,
        // precomputed alpha, positions of both endpoints) is read via SSBOs bound in
        // `scene.bind_group()` — no vertex or index buffer needed for this pipeline.
        if scene.num_edges() > 0 {
            pass.set_pipeline(edges_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, scene.bind_group(), &[]);
            pass.draw(0..4, 0..scene.num_edges());
        }

        pass.set_pipeline(points_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(1, scene.bind_group(), &[]);
        pass.set_vertex_buffer(0, scene.positions_buffer().slice(..));
        pass.draw(0..6, 0..scene.num_points());

        // Text quads last — screen-anchored overlays draw over everything. The rects
        // were written in this fn's prelude, before this encoder submits.
        if !texts.is_empty() {
            let text_pipeline = if resolve_view.is_some() {
                &self.text_pipeline_msaa
            } else {
                &self.text_pipeline_1x
            };
            pass.set_pipeline(text_pipeline);
            for (quad, _) in texts {
                pass.set_bind_group(0, &quad.bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
        }
    }

    /// Draw text quads onto an already-populated single-sample target WITHOUT clearing it
    /// (LoadOp::Load). Used by the density path, whose tone pass has already written the
    /// image and just needs the title/legend composited on top — same pipeline + rect
    /// upload as the scene pass's text draw, so the compositing is identical.
    pub fn draw_texts_load(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        texts: &[(&TextQuad, [f32; 4])],
    ) {
        if texts.is_empty() {
            return;
        }
        for (quad, rect) in texts {
            self.queue
                .write_buffer(&quad.rect_buffer, 0, bytemuck::cast_slice(rect));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("density text pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.text_pipeline_1x);
        for (quad, _) in texts {
            pass.set_bind_group(0, &quad.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

fn build_points_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("points shader"),
        source: wgpu::ShaderSource::Wgsl(
            concat!(
                include_str!("../shaders/srgb.wgsl"),
                include_str!("../shaders/points.wgsl")
            )
            .into(),
        ),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("points pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: (std::mem::size_of::<f32>() * 2) as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

fn build_edges_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("edges shader"),
        source: wgpu::ShaderSource::Wgsl(
            concat!(
                include_str!("../shaders/srgb.wgsl"),
                include_str!("../shaders/lines.wgsl")
            )
            .into(),
        ),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("edges pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            // Each edge instance is a 4-vertex quad; strip order is A−, A+, B−, B+.
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}

fn build_text_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("text shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/text.wgsl").into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("text pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: sample_count,
            ..Default::default()
        },
        multiview: None,
        cache: None,
    })
}
