//! Offscreen PNG export via [`SceneRenderer`].
//!
//! Allocates an off-screen render target at arbitrary resolution, invokes the scene pass
//! (with optional MSAA + resolve), copies the result texture into a mapped buffer, and
//! encodes the pixels as PNG on disk.
//!
//! Format handling: pipelines were built against the surface format (may be
//! `Rgba8UnormSrgb` OR `Bgra8UnormSrgb` depending on platform), so we allocate the
//! offscreen texture with the same format. On BGRA backends we swap the R/B bytes on the
//! CPU side before handing pixels to the `image` crate (which wants RGBA byte order).
//!
//! MSAA path: `sample_count = MSAA_SAMPLE_COUNT` on the render target, hardware resolves
//! to a single-sample texture at end-of-pass, we copy the resolved texture to the buffer.
//! Non-MSAA path: single-sample texture, no resolve.
//!
//! Row alignment: `wgpu` requires `bytes_per_row` in `copy_texture_to_buffer` to be a
//! multiple of 256. We allocate a padded buffer, then strip padding row-by-row into a
//! tight `Vec<u8>` for the PNG encoder.

use std::path::PathBuf;

use anyhow::{Context, Result};
use image::ImageBuffer;

use crate::density::{compute_anchors, DensityRenderer};
use crate::scene::Scene;
use crate::scene_renderer::{SceneRenderer, TextQuad, MSAA_SAMPLE_COUNT};
use crate::settings::DensityParams;

/// wgpu's `copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of this.
const BUFFER_COPY_ROW_ALIGNMENT: u32 = 256;

pub struct ExportConfig {
    pub width: u32,
    pub height: u32,
    /// If true, render into a `MSAA_SAMPLE_COUNT`-sample texture and resolve. Otherwise
    /// direct single-sample render.
    pub msaa: bool,
    /// Clear color for the offscreen target (linear or sRGB depending on format — the
    /// caller is responsible for matching the format's expectation).
    pub clear_color: wgpu::Color,
    /// Output path. Extension `.png` (or anything the `image` crate infers).
    pub path: PathBuf,
}

/// Render the scene at `config.width × config.height` to `config.path`.
///
/// Takes `&mut SceneRenderer` because we push view uniforms before the render pass — the
/// caller has already built the [`View`](crate::framing::View) for the target dimensions
/// (they know about camera + sizing config; the exporter is agnostic). App will overwrite
/// these uniforms on its next tick.
///
/// `texts` are the screen-anchored quads (title, legend — already rasterized at export
/// scale by the caller), each with its NDC rect in the export target — drawn by the same
/// pass as the scene, so the PNG's text compositing is identical to the window's.
pub fn export_png(
    scene_renderer: &mut SceneRenderer,
    scene: &Scene,
    view: &crate::framing::View,
    config: &ExportConfig,
    texts: &[(&crate::scene_renderer::TextQuad, [f32; 4])],
) -> Result<()> {
    scene_renderer.set_view(view);

    let device = scene_renderer.device().clone();
    let queue = scene_renderer.queue().clone();
    let format = scene_renderer.target_format();

    // Single-sample "resolve" texture — always allocated because this is what we read
    // back to CPU. In the MSAA path, the hardware resolve writes here at end-of-pass.
    let resolve_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("export resolve target"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let resolve_view = resolve_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // MSAA texture (if requested) — pipelines rendered here at MSAA_SAMPLE_COUNT samples.
    let msaa_view_holder = if config.msaa {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export msaa target"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Keep the texture alive by binding to a local (view holds a reference internally
        // but the texture handle must outlive the render pass).
        Some((tex, view))
    } else {
        None
    };

    // Buffer sized for CPU readback with wgpu's row-alignment requirement.
    let bytes_per_pixel: u32 = 4;
    let unpadded_bytes_per_row = config.width * bytes_per_pixel;
    let padded_bytes_per_row = padded_row_bytes(unpadded_bytes_per_row);
    let buffer_size = (padded_bytes_per_row as u64) * (config.height as u64);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("export readback buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("export encoder"),
    });

    // Render pass. MSAA path passes the multisample view as target and the resolve view
    // as the resolve target; single-sample path passes the resolve view directly.
    if let Some((_tex, msaa_view)) = &msaa_view_holder {
        scene_renderer.render(
            &mut encoder,
            msaa_view,
            Some(&resolve_view),
            scene,
            config.clear_color,
            texts,
        );
    } else {
        scene_renderer.render(
            &mut encoder,
            &resolve_view,
            None,
            scene,
            config.clear_color,
            texts,
        );
    }

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resolve_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(config.height),
            },
        },
        wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
    );

    let submission_index = queue.submit(std::iter::once(encoder.finish()));

    // Map + wait. `PollType::WaitForSubmissionIndex` blocks until the just-submitted work
    // finishes, at which point map_async's callback fires.
    let buffer_slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::WaitForSubmissionIndex(submission_index))
        .context("wgpu device poll while waiting for export submission")?;
    receiver
        .recv()
        .context("map_async result never sent")?
        .context("buffer map failed")?;

    // Strip row padding into a tight Vec<u8>.
    let mapped = buffer_slice.get_mapped_range();
    let mut pixels =
        Vec::with_capacity((unpadded_bytes_per_row as usize) * (config.height as usize));
    for row in 0..config.height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    buffer.unmap();

    // If the surface format is BGRA, swap the R and B bytes so `image` gets RGBA order.
    if is_bgra_format(format) {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }

    let img = ImageBuffer::<image::Rgba<u8>, _>::from_raw(config.width, config.height, pixels)
        .context("failed to build ImageBuffer from raw pixels — dimensions mismatch")?;
    img.save(&config.path)
        .with_context(|| format!("PNG save to {}", config.path.display()))?;

    Ok(())
}

/// Density-mode PNG export: accumulate → blur → readback → percentile anchors → tonemap →
/// text, all offscreen at `config.width × config.height`. No MSAA — accumulation is additive
/// coverage, not edge geometry, so an MSAA resolve would average away the overdraw the
/// heatmap depends on (the analytic AA in the accum shader handles edges instead).
///
/// The caller must have already pushed the view + channel uniforms onto `scene_renderer`
/// (its `@group(0)` bind group is what the accumulation reads). `ramp` comes from the
/// active theme/config (stop 0 is the canvas color); `texts` are the export-scale
/// title/legend quads.
///
/// Two submissions: the first runs accumulate+blur+copy so we can read the density back and
/// compute anchors on the CPU (a percentile pass needs the finished field); the second
/// tone-maps + draws text + copies the LDR image out. Deterministic — fixed sequence.
#[allow(clippy::too_many_arguments)]
pub fn export_png_density(
    scene_renderer: &mut SceneRenderer,
    density: &mut DensityRenderer,
    scene: &Scene,
    view: &crate::framing::View,
    params: &DensityParams,
    ramp: &[[u8; 4]],
    config: &ExportConfig,
    texts: &[(&TextQuad, [f32; 4])],
) -> Result<()> {
    scene_renderer.set_view(view);
    let device = scene_renderer.device().clone();
    let queue = scene_renderer.queue().clone();
    let format = scene_renderer.target_format();

    density.ensure_lut(ramp);

    // ---- submission 1: accumulate + blur + copy density to a readback buffer ----
    let mut enc1 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("density accumulate encoder"),
    });
    let readback = density.accumulate_and_readback(
        &mut enc1,
        scene,
        scene_renderer.uniform_bind_group(),
        config.width,
        config.height,
        params.sigma,
    );
    let idx = queue.submit(std::iter::once(enc1.finish()));
    let field = readback.read(&device, idx)?;
    let anchors = compute_anchors(params, &field);
    println!(
        "headless: density anchors knee={:.4} white={:.4} (lit-pixel percentiles)",
        anchors.knee, anchors.white
    );

    // ---- submission 2: tonemap → LDR target, text on top, copy out ----
    let ldr_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("density ldr target"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let ldr_view = ldr_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let unpadded_bytes_per_row = config.width * 4;
    let padded_bytes_per_row = padded_row_bytes(unpadded_bytes_per_row);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density export readback"),
        size: (padded_bytes_per_row as u64) * (config.height as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("density tonemap encoder"),
    });
    density.tonemap(&mut enc2, &ldr_view, anchors, &readback);
    // FUTURE overlay slot: a crisp-linework pass would draw here, between tonemap and text.
    scene_renderer.draw_texts_load(&mut enc2, &ldr_view, texts);
    enc2.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &ldr_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(config.height),
            },
        },
        wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
    );
    let idx2 = queue.submit(std::iter::once(enc2.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::WaitForSubmissionIndex(idx2))
        .context("poll for density LDR readback")?;
    rx.recv()
        .context("density LDR map never sent")?
        .context("density LDR map failed")?;
    let mapped = slice.get_mapped_range();
    let mut pixels =
        Vec::with_capacity((unpadded_bytes_per_row as usize) * (config.height as usize));
    for row in 0..config.height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&mapped[start..end]);
    }
    drop(mapped);
    buffer.unmap();
    if is_bgra_format(format) {
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    let img = ImageBuffer::<image::Rgba<u8>, _>::from_raw(config.width, config.height, pixels)
        .context("failed to build density ImageBuffer — dimensions mismatch")?;
    img.save(&config.path)
        .with_context(|| format!("density PNG save to {}", config.path.display()))?;
    Ok(())
}

fn padded_row_bytes(unpadded: u32) -> u32 {
    unpadded.div_ceil(BUFFER_COPY_ROW_ALIGNMENT) * BUFFER_COPY_ROW_ALIGNMENT
}

fn is_bgra_format(format: wgpu::TextureFormat) -> bool {
    matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    )
}
