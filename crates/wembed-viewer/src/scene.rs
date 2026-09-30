//! Per-vertex and per-edge GPU buffers for the rendered graph.
//!
//! Scene owns the topology-scale wgpu resources: positions (vertex/instance
//! buffer), edges (index buffer), and the per-vertex storage buffers (colors,
//! highlight mask) exposed as `@group(1)` in the shaders. Renderer owns wgpu
//! init + pipelines + per-frame uniforms as `@group(0)`.
//!
//! The split exists so adding a new per-vertex attribute (per-vertex size,
//! per-vertex flag, ...) is one file's worth of change — this one — plus a
//! shader binding. Previously the same change touched Renderer's field list,
//! its bind-group layout, its bind-group builder, its reload path, and added
//! a shallow `set_X(&[T])` method.
//!
//! Silent-fallback for colors — "if the file length didn't match current N,
//! ignore" — is enforced upstream by `Attributes::colors_for(n)`. Scene trusts
//! the caller: `Topology.colors = Some(cs)` implies `cs.len() ==
//! positions.len() / 2`. Highlight mask always fills to `num_points`.

use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// Placeholder color for the colors storage buffer when the caller didn't
/// supply per-vertex colors. Little-endian bytes = `[R=0x80, G=0x80, B=0x80,
/// A=0xff]` — opaque medium gray, matches what the shader draws in default
/// (non-colored) mode.
const DUMMY_COLOR: u32 = 0xff80_8080;

/// Placeholder for the per-edge colors buffer when no `--edge-colors` file
/// matched the edge count. Opaque white (`[R,G,B,A] = [ff,ff,ff,ff]`) — never
/// actually drawn, because App/headless downgrade the `file` edge-color source
/// to `theme` whenever the buffer is unpopulated (so the shader's file branch
/// only runs against real data).
const DUMMY_EDGE_COLOR: u32 = 0xffff_ffff;

/// Default highlight-mask entry — `1` = bright (no fade). Written on new / on
/// reload so the scene renders normally before Highlight populates the mask.
const MASK_BRIGHT: u32 = 1;

/// Default per-vertex size multiplier when the embedder didn't provide weights
/// (or on the fallback path). `1.0` = base radius, unmodified. The shader
/// reads this only when the size-by-weight channel is enabled — so the fallback
/// buffer effectively means "size-by-weight is a no-op" when engaged.
const VALUE_UNIT: f32 = 1.0;

/// Everything Scene needs to build or reload itself, in one struct. Mirrors
/// the embedder's `Update::Topology` publish so wiring from App is direct.
///
/// Colors is `Option` because a `--colors` file may not have been passed OR
/// its length may not match current N (layered coarser layers). App consults
/// `Attributes::colors_for(n)` — Scene trusts the length invariant.
pub struct Topology<'a> {
    /// Flat interleaved `[x0, y0, x1, y1, ...]`. Length must be even; sets
    /// `num_points = positions.len() / 2`.
    pub positions: &'a [f32],
    /// Undirected edges as `[src, dst]` vertex-id pairs.
    pub edges: &'a [[u32; 2]],
    /// Per-vertex packed RGBA8 colors. `None` → medium-gray fallback.
    pub colors: Option<&'a [u32]>,
    /// Per-vertex size multipliers, one f32 per vertex. From
    /// `embedder::Topology::size_multipliers` (already preprocessed as
    /// `sqrt(w_i / mean_w)`). `None` → uniform 1.0 fallback (size-by-weight
    /// becomes a no-op when engaged).
    pub size_multipliers: Option<&'a [f32]>,
    /// Per-edge alpha multiplier `1 / (w_a * w_b)` (raw weights, no
    /// normalization). Length == `edges.len()`. Consumed by the lines shader
    /// as a per-edge constant (no interpolation across the segment). `None`
    /// → uniform 1.0 fallback (edge-alpha-by-weight becomes a no-op).
    pub edge_alphas: Option<&'a [f32]>,
    /// Per-edge baked colors (packed RGBA8) from a `--edge-colors` file, used
    /// by the `file` edge-color source. `None` → opaque-white fallback (never
    /// drawn — the source is downgraded to `theme` upstream when absent).
    /// Length, when `Some`, == `edges.len()`.
    pub edge_colors: Option<&'a [u32]>,
}

pub struct Scene {
    // Per-vertex buffers, all sized to `num_points`. Positions doubles as an
    // SSBO for the lines shader (VERTEX | STORAGE usage) — points uses it as
    // a VBO via `set_vertex_buffer`, lines reads it as a storage array.
    positions_buffer: wgpu::Buffer,
    colors_buffer: wgpu::Buffer,
    mask_buffer: wgpu::Buffer,
    values_buffer: wgpu::Buffer,
    /// Per-vertex depth-shade values in [0, 1] (projection residual, quantile-normalized).
    /// Zero-initialized — stays all-0 (no fade) at dimension 2; updated via `write_depth`
    /// whenever the projection refits at dimension > 2.
    depth_buffer: wgpu::Buffer,
    // Per-edge buffers, sized to `num_edges`. Both accessed as SSBOs by the
    // lines shader (indexed by `@builtin(instance_index)` in the instanced
    // draw). Always present with a 1-element sentinel when `num_edges == 0`
    // so the bind group is always valid.
    edges_buffer: wgpu::Buffer,
    edge_alphas_buffer: wgpu::Buffer,
    edge_colors_buffer: wgpu::Buffer,
    /// Per-edge stress ratios `|pos_a − pos_b| / (w_a·w_b)^(1/dim)` for the `length`
    /// edge-color source, computed CPU-side over the FULL d-dim positions. Initialized
    /// to the neutral 1.0; updated via `write_edge_lengths` while the source is active.
    edge_lengths_buffer: wgpu::Buffer,

    /// Binds colors + mask + values + positions + edges + edge_alphas +
    /// edge_colors + depth + edge_lengths as `@group(1) @binding({0..8})`. Rebuilt on
    /// reload because the underlying buffers get replaced on topology changes.
    bind_group: wgpu::BindGroup,

    num_points: u32,
    /// Number of edges (undirected). Draw call issues `pass.draw(0..2,
    /// 0..num_edges)` — 2 vertices per instance, `num_edges` instances.
    num_edges: u32,
}

/// Preflight: verify the scene's biggest buffers fit the device's granted limits BEFORE
/// creating them, so an oversized graph fails with one readable line instead of a wgpu
/// validation panic mid-frame. Positions (8 B/vertex) and edges (8 B/edge) are the two
/// largest bindings; every other scene buffer is 4 B per vertex or edge, so checking
/// these two covers all six.
pub fn check_device_limits(
    device: &wgpu::Device,
    num_vertices: usize,
    num_edges: usize,
) -> anyhow::Result<()> {
    let limits = device.limits();
    let cap = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
    for (name, count) in [("vertex", num_vertices), ("edge", num_edges)] {
        let bytes = (count as u64) * 8;
        if bytes > cap {
            anyhow::bail!(
                "graph too large for this GPU: the {name} buffer needs {:.1} MiB \
                 ({count} {name}s × 8 B) but the device caps buffer bindings at {:.1} MiB",
                bytes as f64 / (1024.0 * 1024.0),
                cap as f64 / (1024.0 * 1024.0),
            );
        }
    }
    Ok(())
}

/// Bind-group-layout descriptor for the scene's per-vertex storage buffers.
/// Static because the shape doesn't depend on data — Renderer::new consults
/// this to build pipelines up-front before any Scene exists.
///
/// Bindings live in `@group(1)`; `@group(0)` is uniforms owned by Renderer.
pub fn bind_group_layout_desc() -> wgpu::BindGroupLayoutDescriptor<'static> {
    const SSBO_ENTRY: wgpu::BindGroupLayoutEntry = wgpu::BindGroupLayoutEntry {
        binding: 0,
        // Vertex-only: fragment shaders receive per-vertex data via
        // interpolated varyings, not direct SSBO reads.
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    const ENTRIES: &[wgpu::BindGroupLayoutEntry] = &[
        // 0 = colors, 1 = highlight mask, 2 = per-vertex size multipliers,
        // 3 = positions (SSBO view for the lines shader — points still uses
        //     the same buffer as a VBO), 4 = edges (u32 pairs), 5 = per-edge
        //     alpha multipliers, 6 = per-edge baked colors (packed RGBA8),
        //     7 = per-vertex depth-shade values, 8 = per-edge stress ratios.
        SSBO_ENTRY,
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 5,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            ..SSBO_ENTRY
        },
        wgpu::BindGroupLayoutEntry {
            binding: 8,
            ..SSBO_ENTRY
        },
    ];
    wgpu::BindGroupLayoutDescriptor {
        label: Some("scene bind group layout"),
        entries: ENTRIES,
    }
}

impl Scene {
    /// Build the initial scene. Caller retains ownership of `topo`; Scene
    /// copies what it needs into GPU buffers.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, topo: &Topology) -> Self {
        assert!(
            topo.positions.len().is_multiple_of(2),
            "positions must be xy-interleaved"
        );
        let num_points = (topo.positions.len() / 2) as u32;
        let num_edges = topo.edges.len() as u32;

        let positions_buffer = build_positions_buffer(device, topo.positions);
        let colors_buffer = build_colors_buffer(device, num_points, topo.colors);
        let mask_buffer = build_mask_buffer(device, num_points);
        let values_buffer = build_values_buffer(device, num_points, topo.size_multipliers);
        let depth_buffer = build_depth_buffer(device, num_points);
        let edges_buffer = build_edges_buffer(device, topo.edges);
        let edge_alphas_buffer = build_edge_alphas_buffer(device, num_edges, topo.edge_alphas);
        let edge_colors_buffer = build_edge_colors_buffer(device, num_edges, topo.edge_colors);
        let edge_lengths_buffer = build_edge_lengths_buffer(device, num_edges);
        let bind_group = build_bind_group(
            device,
            layout,
            &colors_buffer,
            &mask_buffer,
            &values_buffer,
            &positions_buffer,
            &edges_buffer,
            &edge_alphas_buffer,
            &edge_colors_buffer,
            &depth_buffer,
            &edge_lengths_buffer,
        );

        Self {
            positions_buffer,
            colors_buffer,
            mask_buffer,
            values_buffer,
            depth_buffer,
            edges_buffer,
            edge_alphas_buffer,
            edge_colors_buffer,
            edge_lengths_buffer,
            bind_group,
            num_points,
            num_edges,
        }
    }

    /// Hot-path: replace positions in-place. Length MUST equal `num_points *
    /// 2`. No allocation, no rebind — just a `queue.write_buffer`. Called
    /// every frame while the embedder is running.
    pub fn write_positions(&self, queue: &wgpu::Queue, positions: &[f32]) {
        debug_assert_eq!(positions.len(), (self.num_points as usize) * 2);
        queue.write_buffer(&self.positions_buffer, 0, cast_slice(positions));
    }

    /// Push a new highlight mask. Length MUST equal `num_points`. Same shape
    /// as `write_positions` — no realloc, no rebind. Called on hover/search
    /// state changes (rare relative to frames).
    pub fn write_mask(&self, queue: &wgpu::Queue, mask: &[u32]) {
        debug_assert_eq!(mask.len(), self.num_points as usize);
        queue.write_buffer(&self.mask_buffer, 0, cast_slice(mask));
    }

    /// Push a new colors buffer. Length MUST equal `num_points`. Used when the theme
    /// changes at runtime — App resolves theme accents → per-vertex packed RGBA and
    /// uploads via this method without rebuilding the whole scene.
    pub fn write_colors(&self, queue: &wgpu::Queue, colors: &[u32]) {
        debug_assert_eq!(colors.len(), self.num_points as usize);
        queue.write_buffer(&self.colors_buffer, 0, cast_slice(colors));
    }

    /// Push new depth-shade values. Length MUST equal `num_points`. Same shape as
    /// `write_positions`; called per positions update at dimension > 2.
    pub fn write_depth(&self, queue: &wgpu::Queue, depth: &[f32]) {
        debug_assert_eq!(depth.len(), self.num_points as usize);
        queue.write_buffer(&self.depth_buffer, 0, cast_slice(depth));
    }

    /// Push new per-edge stress ratios. Length MUST equal `num_edges`. Called per
    /// positions update while the `length` edge-color source is active.
    pub fn write_edge_lengths(&self, queue: &wgpu::Queue, ratios: &[f32]) {
        debug_assert_eq!(ratios.len(), self.num_edges as usize);
        if !ratios.is_empty() {
            queue.write_buffer(&self.edge_lengths_buffer, 0, cast_slice(ratios));
        }
    }

    /// Rebuild all buffers for a topology change — initial publish (already
    /// handled by `new`), or a layered-embedder layer transition. Vertex
    /// count generally differs, so every buffer is reallocated and the bind
    /// group is rebuilt to reference the new backing storage.
    ///
    /// Mask resets to all-bright. `Highlight::set_topology` resets its own
    /// tracking sets in parallel, so the mask and Highlight stay coherent
    /// without cross-module plumbing.
    pub fn reload(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        topo: &Topology,
    ) {
        assert!(
            topo.positions.len().is_multiple_of(2),
            "positions must be xy-interleaved"
        );
        self.num_points = (topo.positions.len() / 2) as u32;
        self.num_edges = topo.edges.len() as u32;

        self.positions_buffer = build_positions_buffer(device, topo.positions);
        self.colors_buffer = build_colors_buffer(device, self.num_points, topo.colors);
        self.mask_buffer = build_mask_buffer(device, self.num_points);
        self.values_buffer = build_values_buffer(device, self.num_points, topo.size_multipliers);
        self.depth_buffer = build_depth_buffer(device, self.num_points);
        self.edges_buffer = build_edges_buffer(device, topo.edges);
        self.edge_alphas_buffer =
            build_edge_alphas_buffer(device, self.num_edges, topo.edge_alphas);
        self.edge_colors_buffer =
            build_edge_colors_buffer(device, self.num_edges, topo.edge_colors);
        self.edge_lengths_buffer = build_edge_lengths_buffer(device, self.num_edges);
        self.bind_group = build_bind_group(
            device,
            layout,
            &self.colors_buffer,
            &self.mask_buffer,
            &self.values_buffer,
            &self.positions_buffer,
            &self.edges_buffer,
            &self.edge_alphas_buffer,
            &self.edge_colors_buffer,
            &self.depth_buffer,
            &self.edge_lengths_buffer,
        );
    }

    // ---- render-pass surface ----------------------------------------------

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn positions_buffer(&self) -> &wgpu::Buffer {
        &self.positions_buffer
    }

    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    /// Number of edges. The lines pipeline draws these as `num_edges`
    /// instances of a 2-vertex line segment (`pass.draw(0..2, 0..num_edges)`).
    /// The whole edge pass is skipped when this is zero.
    pub fn num_edges(&self) -> u32 {
        self.num_edges
    }
}

// ---- buffer builders -------------------------------------------------------

fn build_positions_buffer(device: &wgpu::Device, positions: &[f32]) -> wgpu::Buffer {
    // VERTEX for the points pipeline (VBO via `set_vertex_buffer`), STORAGE for
    // the lines pipeline (SSBO — lines reads positions[graph_vid] indirectly
    // via the edges buffer, since indexed drawing can't express per-line-vertex
    // attributes).
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("positions"),
        contents: cast_slice(positions),
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
    })
}

/// Edges as a flat `[[src, dst], ...]` SSBO, indexed by `instance_index` in
/// the instanced draw. Always sized to at least 1 pair so the bind group is
/// well-defined even when the graph has no edges.
fn build_edges_buffer(device: &wgpu::Device, edges: &[[u32; 2]]) -> wgpu::Buffer {
    let contents: Vec<u32> = if edges.is_empty() {
        vec![0, 0]
    } else {
        edges.iter().flatten().copied().collect()
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edges"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Per-edge alpha multiplier. Always sized to `num_edges.max(1)` so the bind
/// group is well-defined even for the layered embedder's coarsest layers
/// (which can have 1 node and 0 edges). `None` OR an empty input slice falls
/// back to a `1.0` sentinel — edge-alpha-by-weight becomes a no-op in that
/// case, which is fine because the corresponding draw is skipped when
/// `num_edges == 0`.
fn build_edge_alphas_buffer(
    device: &wgpu::Device,
    num_edges: u32,
    edge_alphas: Option<&[f32]>,
) -> wgpu::Buffer {
    let n = (num_edges as usize).max(1);
    let contents: Vec<f32> = match edge_alphas {
        Some(alphas) if !alphas.is_empty() => alphas.to_vec(),
        _ => vec![1.0; n],
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge alphas"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Per-edge baked colors (packed RGBA8). Always sized to `num_edges.max(1)` so
/// the bind group is well-defined even for the layered embedder's coarsest
/// layers. `None` OR an empty slice → opaque-white fallback; the caller
/// downgrades the `file` edge-color source to `theme` in that case, so the
/// fallback bytes are never actually read by the shader's file branch.
fn build_edge_colors_buffer(
    device: &wgpu::Device,
    num_edges: u32,
    edge_colors: Option<&[u32]>,
) -> wgpu::Buffer {
    let n = (num_edges as usize).max(1);
    let contents: Vec<u32> = match edge_colors {
        Some(cs) if !cs.is_empty() => cs.to_vec(),
        _ => vec![DUMMY_EDGE_COLOR; n],
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge colors"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Always sized to `num_points` (or 1 if the scene is empty) so any in-range
/// instance-index read is well-defined at the buffer level.
fn build_colors_buffer(
    device: &wgpu::Device,
    num_points: u32,
    colors: Option<&[u32]>,
) -> wgpu::Buffer {
    let n = (num_points as usize).max(1);
    let contents: Vec<u32> = match colors {
        Some(cs) => cs.to_vec(),
        None => vec![DUMMY_COLOR; n],
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("colors"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn build_mask_buffer(device: &wgpu::Device, num_points: u32) -> wgpu::Buffer {
    let n = (num_points as usize).max(1);
    let contents = vec![MASK_BRIGHT; n];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("highlight mask"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Per-edge stress ratios, initialized to the neutral 1.0 (colormap midpoint) so the
/// `length` source renders sanely before the first CPU upload. Sized to
/// `num_edges.max(1)` like the other per-edge buffers.
fn build_edge_lengths_buffer(device: &wgpu::Device, num_edges: u32) -> wgpu::Buffer {
    let n = (num_edges as usize).max(1);
    let contents = vec![1.0_f32; n];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge stress ratios"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Per-vertex depth-shade values, zero-initialized (= no fade). Sized to
/// `num_points.max(1)` like the other per-vertex buffers; content arrives via
/// `write_depth` only when a dimension > 2 projection is active.
fn build_depth_buffer(device: &wgpu::Device, num_points: u32) -> wgpu::Buffer {
    let n = (num_points as usize).max(1);
    let contents = vec![0.0_f32; n];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("depth shade"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

/// Always sized to `num_points`. `None` → all-1.0 fallback (matches the
/// "size-by-weight becomes a no-op" invariant when weights are absent).
fn build_values_buffer(
    device: &wgpu::Device,
    num_points: u32,
    values: Option<&[f32]>,
) -> wgpu::Buffer {
    let n = (num_points as usize).max(1);
    let contents: Vec<f32> = match values {
        Some(vs) => vs.to_vec(),
        None => vec![VALUE_UNIT; n],
    };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("size multipliers"),
        contents: cast_slice(&contents),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    colors: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    values: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    edges: &wgpu::Buffer,
    edge_alphas: &wgpu::Buffer,
    edge_colors: &wgpu::Buffer,
    depth: &wgpu::Buffer,
    edge_lengths: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mask.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: positions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: edges.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: edge_alphas.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: edge_colors.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: depth.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: edge_lengths.as_entire_binding(),
            },
        ],
    })
}
