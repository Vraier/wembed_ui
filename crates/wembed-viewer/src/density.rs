//! Density (heatmap) render strategy: accumulate → blur → normalize + colormap.
//!
//! An alternate to the normal points/edges pass. Edges and points splat their scalar weight
//! into an R16Float accumulation texture with additive blending (no MSAA); a separable
//! gaussian smooths it (post-blur ≡ per-splat gaussian by linearity); then a log tone-curve
//! with an OKLab ramp maps density → color into the sRGB LDR target. The text/legend pass
//! draws on top afterwards, unchanged.
//!
//! Determinism: headless runs a single fixed sequence — accumulate, blur, read back the
//! density, compute percentile anchors on the CPU, build the LUT, tone-map. No MSAA (the
//! accumulation is additive coverage, not edge geometry — resolve would average away the
//! very overdraw we want), so there's nothing resolution-dependent to jitter.
//!
//! Phase-2 hooks left in place: the tone-map pass is a distinct step so a crisp-linework
//! overlay can slot between it and the text pass; `Curve` is an enum so histogram-EQ can be
//! added; the accumulation format is centralized (`ACCUM_FORMAT`) so a color-preserving
//! RGBA16F variant is a one-constant change plus a wider accumulation shader.
//!
//! Accumulation format note: the spec calls for R32Float, but wgpu 25 marks R32Float
//! non-blendable (additive blend is rejected by validation), so this uses R16Float — the
//! only float format that additively blends without unavailable native features. The tone
//! map normalizes by percentiles, so only *relative* density matters; R16Float's ~3-digit
//! relative precision is ample for a monotone log curve.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::scene::Scene;
use crate::settings::DensityParams;
use crate::theme::{linear_to_srgb, srgb_to_linear};

/// Accumulation + blur texture format. R16Float is blendable + filterable in core WebGPU;
/// R32Float is not blendable in wgpu 25 (see module docs).
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// LUT width — 512 fused (curve + ramp) entries is smooth enough that banding is invisible
/// against alpha-blended density, and cheap to rebuild live on a slider drag.
const LUT_SIZE: u32 = 512;

/// Resolved, render-ready density anchors + curve, produced by [`compute_anchors`] from the
/// density histogram (percentile anchors) or straight from explicit config values.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DensityAnchors {
    pub knee: f32,
    pub white: f32,
}

/// Map density → t∈[0,1] on the log curve. MUST match `curve_t` in `density_tonemap.wgsl`
/// and the LUT's t-axis. `scale` is a small knee-relative floor so the log doesn't blow up
/// just above the knee; chosen as a fraction of the (white−knee) span. Phase-2 histogram-EQ
/// would add a sibling curve here + a `curve` selector; this is the only one for now.
/// (Kept even though only the test calls it directly — it's the canonical Rust mirror of the
/// shader's `curve_t`, so drift between the two is caught by `curve_t_anchors_map_to_*`.)
#[allow(dead_code)]
pub fn curve_t(d: f32, knee: f32, white: f32) -> f32 {
    let (scale, inv_log_span) = log_curve_params(knee, white);
    let x = (d - knee).max(0.0);
    ((1.0 + x / scale).ln() * inv_log_span).clamp(0.0, 1.0)
}

/// `(scale, inv_log_span)` for the log curve so `curve_t(white) == 1`. `scale` = 1/16 of the
/// span keeps the curve gentle near the knee and steep toward white; the normalizer inverts
/// `log1p((white-knee)/scale)`.
fn log_curve_params(knee: f32, white: f32) -> (f32, f32) {
    let span = (white - knee).max(1e-6);
    let scale = (span / 16.0).max(1e-6);
    let denom = (1.0 + span / scale).ln().max(1e-6);
    (scale, 1.0 / denom)
}

// ---- OKLab ramp + LUT ------------------------------------------------------

/// linear sRGB → OKLab (Björn Ottosson's transform).
fn linear_srgb_to_oklab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();
    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// OKLab → linear sRGB (inverse of [`linear_srgb_to_oklab`]).
fn oklab_to_linear_srgb(l: f32, a: f32, b: f32) -> [f32; 3] {
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// Interpolate a ramp of sRGB stops in OKLab at parameter `t∈[0,1]`, equal-t spacing of
/// stops. Returns sRGB bytes.
fn ramp_sample_oklab(stops_oklab: &[[f32; 3]], t: f32) -> [u8; 4] {
    let n = stops_oklab.len();
    debug_assert!(n >= 2);
    let t = t.clamp(0.0, 1.0);
    let pos = t * (n - 1) as f32;
    let i = (pos.floor() as usize).min(n - 2);
    let f = pos - i as f32;
    let a = stops_oklab[i];
    let b = stops_oklab[i + 1];
    let lab = [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
    ];
    let lin = oklab_to_linear_srgb(lab[0], lab[1], lab[2]);
    [
        linear_to_srgb(lin[0]),
        linear_to_srgb(lin[1]),
        linear_to_srgb(lin[2]),
        0xff,
    ]
}

/// Build the [`LUT_SIZE`]-entry RGBA8 LUT: entry i = ramp color at t = i/(LUT_SIZE-1),
/// interpolated in OKLab. The tone CURVE lives in the tone-map shader (density→t); the LUT
/// is purely t→color, so the curve and ramp compose without the LUT needing rebuild when
/// anchors change — only a ramp/theme change rebuilds it.
pub fn build_lut(ramp: &[[u8; 4]]) -> Vec<u8> {
    let stops_oklab: Vec<[f32; 3]> = ramp
        .iter()
        .map(|c| {
            linear_srgb_to_oklab(
                srgb_to_linear(c[0]),
                srgb_to_linear(c[1]),
                srgb_to_linear(c[2]),
            )
        })
        .collect();
    let mut out = Vec::with_capacity((LUT_SIZE as usize) * 4);
    for i in 0..LUT_SIZE {
        let t = i as f32 / (LUT_SIZE - 1) as f32;
        out.extend_from_slice(&ramp_sample_oklab(&stops_oklab, t));
    }
    out
}

/// The effective ramp: the config's stops, or a theme bg→accent→fg fallback. Stop 0 is the
/// density render's canvas color (there is no separate background — see the tonemap shader).
pub fn effective_ramp(
    params: &DensityParams,
    background: [u8; 4],
    accent: Option<[u8; 4]>,
    foreground: [u8; 4],
) -> Vec<[u8; 4]> {
    if params.ramp.len() >= 2 {
        params.ramp.clone()
    } else {
        match accent {
            Some(a) => vec![background, a, foreground],
            None => vec![background, foreground],
        }
    }
}

/// Fallback-ramp midpoint: the theme's most chromatic accent. A bare bg→fg ramp is a pure
/// lightness wedge — one hue waypoint at mid densities makes tiers readable.
pub fn pick_ramp_accent(accents: &[[u8; 4]]) -> Option<[u8; 4]> {
    fn chroma(c: [u8; 4]) -> f32 {
        let lab = linear_srgb_to_oklab(
            srgb_to_linear(c[0]),
            srgb_to_linear(c[1]),
            srgb_to_linear(c[2]),
        );
        (lab[1] * lab[1] + lab[2] * lab[2]).sqrt()
    }
    accents
        .iter()
        .copied()
        .max_by(|a, b| chroma(*a).total_cmp(&chroma(*b)))
}

// ---- anchors from the density histogram ------------------------------------

/// Compute render anchors from the read-back density field. Percentile anchors are taken
/// over LIT pixels only (density > 0) — a mostly-empty frame is the norm, so including the
/// zeros would drag every percentile to 0. Explicit config values pass straight through.
pub fn compute_anchors(params: &DensityParams, density: &[f32]) -> DensityAnchors {
    // Lit pixels only: 60–75% of a typical frame is empty background.
    let mut lit: Vec<f32> = density.iter().copied().filter(|&d| d > 0.0).collect();
    let knee = match params.knee {
        Ok(v) => v,
        Err(p) => percentile(&mut lit, p),
    };
    let white = match params.white {
        Ok(v) => v,
        Err(p) => percentile(&mut lit, p),
    };
    // Guard degenerate anchors (uniform field / knee ≥ white) so the curve stays monotone.
    let white = white.max(knee * 1.0001 + 1e-6);
    DensityAnchors { knee, white }
}

/// `p`-th percentile (0..100) of `values`, sorting in place. Empty → 0. Linear-index pick
/// (nearest-rank-ish, clamped) — exact interpolation isn't worth it for an 8-bit-ramp anchor.
fn percentile(values: &mut [f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let frac = (p / 100.0).clamp(0.0, 1.0);
    let idx = (frac * (values.len() - 1) as f32).round() as usize;
    values[idx.min(values.len() - 1)]
}

// ---- GPU uniforms ----------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlurUniforms {
    axis: [f32; 2],
    radius: i32,
    inv_two_sigma_sq: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ToneUniforms {
    knee: f32,
    white: f32,
    inv_log_span: f32,
    scale: f32,
}

/// GPU-side density strategy: owns the density pipelines + a per-size texture set, lazily
/// created (only when density mode is active) so normal rendering allocates nothing extra.
/// The accumulation reuses the scene's `@group(1)` SSBOs and the normal `@group(0)`
/// uniforms, so it stays in lockstep with the geometry the normal pipelines expand.
pub struct DensityRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    accum_points_pipeline: wgpu::RenderPipeline,
    accum_edges_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    tone_pipeline: wgpu::RenderPipeline,

    blur_bind_layout: wgpu::BindGroupLayout,
    tone_bind_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,

    /// Current ramp (for change detection) + its uploaded LUT texture.
    lut_texture: wgpu::Texture,
    lut_view: wgpu::TextureView,
    lut_ramp: Vec<[u8; 4]>,

    /// Per-size accumulation textures (ping-pong pair). Recreated on resolution change.
    size: (u32, u32),
    accum_a: Option<TextureView2>,
    accum_b: Option<TextureView2>,
}

struct TextureView2 {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl DensityRenderer {
    /// Build the density pipelines against the scene's `@group(0)` uniform layout and
    /// `@group(1)` SSBO layout (so accumulation reads the exact same geometry inputs the
    /// normal pipelines do) and the LDR `target_format` the tone pass writes.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        uniform_bind_layout: &wgpu::BindGroupLayout,
        scene_bind_layout: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let accum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density accum shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/density_accum.wgsl").into()),
        });
        let accum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density accum layout"),
            bind_group_layouts: &[uniform_bind_layout, scene_bind_layout],
            push_constant_ranges: &[],
        });
        // Additive (ONE/ONE) so overlapping splats sum — the whole point of accumulation.
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let accum_target = wgpu::ColorTargetState {
            format: ACCUM_FORMAT,
            blend: Some(additive),
            write_mask: wgpu::ColorWrites::RED,
        };
        let accum_points_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("density accum points"),
                layout: Some(&accum_layout),
                vertex: wgpu::VertexState {
                    module: &accum_shader,
                    entry_point: Some("vs_points"),
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
                    module: &accum_shader,
                    entry_point: Some("fs_points"),
                    targets: &[Some(accum_target.clone())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });
        let accum_edges_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("density accum edges"),
            layout: Some(&accum_layout),
            vertex: wgpu::VertexState {
                module: &accum_shader,
                entry_point: Some("vs_edges"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &accum_shader,
                entry_point: Some("fs_edges"),
                targets: &[Some(accum_target)],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Blur: fullscreen triangle, uniform + src texture.
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density blur shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/density_blur.wgsl").into()),
        });
        let blur_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density blur bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                        // textureLoad, no filtering needed
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density blur layout"),
            bind_group_layouts: &[&blur_bind_layout],
            push_constant_ranges: &[],
        });
        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("density blur pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ACCUM_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::RED,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Tone: fullscreen triangle → LDR target.
        let tone_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("density tonemap shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/density_tonemap.wgsl").into(),
            ),
        });
        let tone_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("density tone bind layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tone_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("density tone layout"),
            bind_group_layouts: &[&tone_bind_layout],
            push_constant_ranges: &[],
        });
        let tone_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("density tone pipeline"),
            layout: Some(&tone_layout),
            vertex: wgpu::VertexState {
                module: &tone_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &tone_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("density sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Placeholder LUT (2-stop grayscale); replaced by the first `ensure_lut`.
        let lut_ramp = vec![[0, 0, 0, 0xff], [0xff, 0xff, 0xff, 0xff]];
        let (lut_texture, lut_view) = upload_lut(&device, &queue, &build_lut(&lut_ramp));

        Self {
            device,
            queue,
            accum_points_pipeline,
            accum_edges_pipeline,
            blur_pipeline,
            tone_pipeline,
            blur_bind_layout,
            tone_bind_layout,
            sampler,
            lut_texture,
            lut_view,
            lut_ramp,
            size: (0, 0),
            accum_a: None,
            accum_b: None,
        }
    }

    /// Rebuild the LUT texture iff `ramp` changed. Cheap no-op on the steady state.
    pub fn ensure_lut(&mut self, ramp: &[[u8; 4]]) {
        if self.lut_ramp == ramp {
            return;
        }
        let (tex, view) = upload_lut(&self.device, &self.queue, &build_lut(ramp));
        self.lut_texture = tex;
        self.lut_view = view;
        self.lut_ramp = ramp.to_vec();
    }

    /// Ensure the ping-pong accumulation textures match `(w, h)`; recreate on change.
    fn ensure_textures(&mut self, w: u32, h: u32) {
        if self.size == (w, h) && self.accum_a.is_some() {
            return;
        }
        self.size = (w, h);
        self.accum_a = Some(make_accum(&self.device, w, h, "density accum A"));
        self.accum_b = Some(make_accum(&self.device, w, h, "density accum B"));
    }

    /// Run accumulate + blur into an offscreen R16Float texture at `(w, h)`, then read the
    /// density field back to the CPU. Returns the density row-major (`w*h`), the caller
    /// computes anchors then calls [`DensityRenderer::tonemap`]. Deterministic — a fixed
    /// pass sequence, no MSAA.
    ///
    /// `scene`/`uniform_bind_group` supply the same geometry inputs the normal pipelines
    /// use; `sigma` is the blur in pixels (0 = skip blur).
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_and_readback(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        uniform_bind_group: &wgpu::BindGroup,
        w: u32,
        h: u32,
        sigma: f32,
    ) -> DensityReadback {
        self.ensure_textures(w, h);
        // accumulate + blur leaves the density in accum_a (H+V is an even ping-pong).
        self.accumulate_blur(encoder, scene, uniform_bind_group, sigma);
        let density_tex = &self.accum_a.as_ref().expect("textures ensured").texture;

        // ---- read back the (blurred) density to a padded buffer ----
        let bpp = 2u32; // R16Float = 2 bytes
        let unpadded = w * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density readback"),
            size: (padded as u64) * (h as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: density_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        DensityReadback {
            buffer,
            w,
            h,
            padded_bytes_per_row: padded,
        }
    }

    fn blur_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        axis: [f32; 2],
        radius: i32,
        inv_two_sigma_sq: f32,
    ) {
        let uniforms = BlurUniforms {
            axis,
            radius,
            inv_two_sigma_sq,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("density blur uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density blur bind"),
            layout: &self.blur_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("density blur pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.blur_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Accumulate + blur into the offscreen texture, then tone-map into `target_view` — the
    /// per-frame interactive path. NO readback (that stalls); uses the caller-cached
    /// `anchors`, which auto-exposure refreshes on a debounce via
    /// [`DensityRenderer::accumulate_and_readback`]. The tonemap clears `target_view` (it IS
    /// the background), so the caller draws egui/text on top afterwards with LoadOp::Load.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        scene: &Scene,
        uniform_bind_group: &wgpu::BindGroup,
        w: u32,
        h: u32,
        sigma: f32,
        anchors: DensityAnchors,
        ramp: &[[u8; 4]],
    ) {
        self.ensure_lut(ramp);
        self.ensure_textures(w, h);
        self.accumulate_blur(encoder, scene, uniform_bind_group, sigma);
        // density lives in accum_a after an even number of blur passes (H+V) or when σ=0
        let density = &self.accum_a.as_ref().expect("textures ensured").view;
        self.tonemap_from(encoder, target_view, density, anchors);
    }

    /// Shared accumulate (edges+points) + separable blur into `accum_a` (final home). Used by
    /// both the interactive per-frame path and the read-back path.
    fn accumulate_blur(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &Scene,
        uniform_bind_group: &wgpu::BindGroup,
        sigma: f32,
    ) {
        let accum = self.accum_a.as_ref().expect("textures ensured");
        let scratch = self.accum_b.as_ref().expect("textures ensured");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("density accumulate"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &accum.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if scene.num_edges() > 0 {
                pass.set_pipeline(&self.accum_edges_pipeline);
                pass.set_bind_group(0, uniform_bind_group, &[]);
                pass.set_bind_group(1, scene.bind_group(), &[]);
                pass.draw(0..4, 0..scene.num_edges());
            }
            pass.set_pipeline(&self.accum_points_pipeline);
            pass.set_bind_group(0, uniform_bind_group, &[]);
            pass.set_bind_group(1, scene.bind_group(), &[]);
            pass.set_vertex_buffer(0, scene.positions_buffer().slice(..));
            pass.draw(0..6, 0..scene.num_points());
        }
        if sigma > 0.0 {
            let radius = (3.0 * sigma).ceil().max(1.0) as i32;
            let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
            self.blur_pass(
                encoder,
                &accum.view,
                &scratch.view,
                [1.0, 0.0],
                radius,
                inv_two_sigma_sq,
            );
            self.blur_pass(
                encoder,
                &scratch.view,
                &accum.view,
                [0.0, 1.0],
                radius,
                inv_two_sigma_sq,
            );
        }
    }

    /// Tone-map the (blurred) density in the accumulation texture into `target_view`,
    /// clearing it first. Uses the anchors computed from the read-back histogram + the
    /// current LUT. Call after [`DensityRenderer::accumulate_and_readback`] on the SAME
    /// size; `readback.final_is_scratch` tells us which ping-pong texture holds the density.
    pub fn tonemap(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        anchors: DensityAnchors,
        _readback: &DensityReadback,
    ) {
        // density always lands in accum_a (accumulate_blur's H+V is an even ping-pong).
        let density = &self.accum_a.as_ref().expect("textures ensured").view;
        self.tonemap_from(encoder, target_view, density, anchors);
    }

    /// Tone-map a specific density view into `target_view`. Shared by the read-back path and
    /// the interactive per-frame path.
    fn tonemap_from(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        density: &wgpu::TextureView,
        anchors: DensityAnchors,
    ) {
        let (scale, inv_log_span) = log_curve_params(anchors.knee, anchors.white);
        let uniforms = ToneUniforms {
            knee: anchors.knee,
            white: anchors.white,
            inv_log_span,
            scale,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("density tone uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("density tone bind"),
            layout: &self.tone_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ubuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(density),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("density tonemap pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Overwrite the whole target — the tone pass IS the background.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.tone_pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Handle to a submitted density read-back — the mapped buffer + geometry to decode the
/// R16Float half-floats into a row-major `Vec<f32>`. Kept separate from the anchor compute
/// so the caller controls when to block on the map.
pub struct DensityReadback {
    buffer: wgpu::Buffer,
    w: u32,
    h: u32,
    padded_bytes_per_row: u32,
}

impl DensityReadback {
    /// Map + decode into a row-major `Vec<f32>` of length `w*h`. Blocks on `device.poll`
    /// against `submission_index` (the submission that ran the accumulate/blur/copy).
    pub fn read(
        &self,
        device: &wgpu::Device,
        submission_index: wgpu::SubmissionIndex,
    ) -> anyhow::Result<Vec<f32>> {
        use anyhow::Context;
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::WaitForSubmissionIndex(submission_index))
            .context("poll for density readback")?;
        rx.recv()
            .context("density map_async never sent")?
            .context("density buffer map failed")?;
        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((self.w as usize) * (self.h as usize));
        for row in 0..self.h {
            let start = (row * self.padded_bytes_per_row) as usize;
            for col in 0..self.w {
                let i = start + (col as usize) * 2;
                let bits = u16::from_le_bytes([mapped[i], mapped[i + 1]]);
                out.push(half_to_f32(bits));
            }
        }
        drop(mapped);
        self.buffer.unmap();
        Ok(out)
    }
}

// ---- helpers ---------------------------------------------------------------

fn make_accum(device: &wgpu::Device, w: u32, h: u32, label: &str) -> TextureView2 {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ACCUM_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    TextureView2 { texture, view }
}

fn upload_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lut: &[u8],
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: LUT_SIZE,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("density lut"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB view so the linear sampler filters in linear space, matching the OKLab bake.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        lut,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(LUT_SIZE * 4),
            rows_per_image: Some(1),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// IEEE-754 half (u16 bits) → f32. Density is always finite non-negative, but handle the
/// full range for safety.
fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let val = if exp == 0 {
        // subnormal
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        // inf/nan → treat as 0 (never expected in density)
        return 0.0;
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_t_anchors_map_to_zero_and_one() {
        let (knee, white) = (2.0, 100.0);
        assert!(curve_t(knee, knee, white).abs() < 1e-5, "knee → 0");
        assert!(
            (curve_t(white, knee, white) - 1.0).abs() < 1e-5,
            "white → 1"
        );
        // below the knee clamps to 0, well above white clamps to 1
        assert_eq!(curve_t(0.0, knee, white), 0.0);
        assert_eq!(curve_t(1e6, knee, white), 1.0);
        // monotone increasing between anchors
        let mid = curve_t(0.5 * (knee + white), knee, white);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn percentile_over_lit_ignores_zeros() {
        let params = DensityParams {
            sigma: 1.0,
            knee: Err(50.0),
            white: Err(100.0),
            ramp: vec![],
        };
        // 6 zeros + [1,2,3,4] — the p50 over LIT must land in [1,4], not 0.
        let density = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let a = compute_anchors(&params, &density);
        assert!(a.knee >= 1.0, "knee over lit pixels, got {}", a.knee);
        assert!(a.white >= a.knee, "white >= knee");
        assert!(
            (a.white - 4.0).abs() < 1e-5,
            "p100 lit = max = 4, got {}",
            a.white
        );
    }

    #[test]
    fn explicit_anchors_pass_through() {
        let params = DensityParams {
            sigma: 0.0,
            knee: Ok(3.0),
            white: Ok(5000.0),
            ramp: vec![],
        };
        let a = compute_anchors(&params, &[0.0, 1.0, 2.0]);
        assert_eq!(a.knee, 3.0);
        assert_eq!(a.white, 5000.0);
    }

    #[test]
    fn lut_endpoints_match_ramp_ends() {
        // A 2-stop black→white ramp: LUT[0] ≈ black, LUT[last] ≈ white.
        let lut = build_lut(&[[0, 0, 0, 0xff], [0xff, 0xff, 0xff, 0xff]]);
        assert_eq!(lut.len(), (LUT_SIZE as usize) * 4);
        assert!(
            lut[0] < 8 && lut[1] < 8 && lut[2] < 8,
            "dark end near black"
        );
        let last = (LUT_SIZE as usize - 1) * 4;
        assert!(
            lut[last] > 247 && lut[last + 1] > 247 && lut[last + 2] > 247,
            "bright end near white"
        );
    }

    #[test]
    fn oklab_round_trips_srgb() {
        for &(r, g, b) in &[
            (0u8, 0u8, 0u8),
            (255, 255, 255),
            (255, 153, 64),
            (11, 14, 20),
        ] {
            let lab = linear_srgb_to_oklab(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b));
            let lin = oklab_to_linear_srgb(lab[0], lab[1], lab[2]);
            let (rr, gg, bb) = (
                linear_to_srgb(lin[0]),
                linear_to_srgb(lin[1]),
                linear_to_srgb(lin[2]),
            );
            assert!((rr as i32 - r as i32).abs() <= 1, "R {r}->{rr}");
            assert!((gg as i32 - g as i32).abs() <= 1, "G {g}->{gg}");
            assert!((bb as i32 - b as i32).abs() <= 1, "B {b}->{bb}");
        }
    }

    #[test]
    fn half_to_f32_known_values() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0x4000), 2.0);
        assert!((half_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }
}
