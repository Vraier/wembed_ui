// Density accumulation: edges + points splat their SCALAR weight into an R16Float target
// with additive (ONE/ONE) blending, no MSAA. RGB of any colors file is IGNORED here — the
// only output is a per-fragment weight = file/vertex alpha × global scale × analytic
// coverage. A later blur + tone-curve turns the summed field into color.
//
// Two entry-point pairs share this module — points (vs_points/fs_points, 6-vert quads) and
// edges (vs_edges/fs_edges, 4-vert strip quads) — mirroring the geometry expansion of
// points.wgsl / lines.wgsl so the density splat lands exactly where the normal dot/line
// would. The uniform block is the SAME 144-byte layout as the normal pipelines (bound as
// @group(0)); only the fields this pass reads are documented.

const SIZE_BY_WEIGHT: u32 = 1u;
const EDGE_ALPHA_BY_WEIGHT: u32 = 1u;

// Faded (mask == 0) contribution multiplier — an interactive search/hover nicety. Headless
// never populates the mask, so every vertex is bright there and the result is unaffected.
const FADE_ALPHA: f32 = 0.05;

struct Uniforms {
    view_proj: mat4x4<f32>,
    point_radius: f32,
    vertex_color_source: u32,
    vertex_size_source: u32,
    edge_alpha_source: u32,
    vertex_size_scale: f32,
    vertex_alpha_scale: f32,
    edge_alpha_scale: f32,
    edge_color_source: u32,
    edge_theme_color: vec4<f32>,
    edge_len_log_min: f32,
    edge_len_log_max: f32,
    depth_fade: f32,
    line_width_px: f32,
    viewport_size: vec2<f32>,
    edge_width_scale: f32,
    _pad1: f32,
    // Unused here, declared for struct-layout parity with `points.wgsl`.
    vertex_theme_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var<storage, read> colors: array<u32>;      // RGB unused; only .a read
@group(1) @binding(1) var<storage, read> mask: array<u32>;
@group(1) @binding(2) var<storage, read> values: array<f32>;      // sqrt(w/mean_w) size mult
@group(1) @binding(3) var<storage, read> positions: array<vec2<f32>>;
@group(1) @binding(4) var<storage, read> edges: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read> edge_alphas: array<f32>;
@group(1) @binding(6) var<storage, read> edge_colors: array<u32>; // only .a read (file alpha)

// ---- points ----------------------------------------------------------------

struct POut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) weight: f32,
};

var<private> QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

@vertex
fn vs_points(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
    @location(0) instance_pos: vec2<f32>,
) -> POut {
    let q = QUAD[vi];
    var size_mult: f32 = 1.0;
    if u.vertex_size_source == SIZE_BY_WEIGHT {
        size_mult = values[ii];
    }
    let world = instance_pos + q * u.point_radius * size_mult * u.vertex_size_scale;
    var out: POut;
    out.clip_pos = u.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.uv = q;
    // Per-vertex weight = file alpha (RGB ignored) × global vertex-alpha scale × fade.
    let a = unpack4x8unorm(colors[ii]).a;
    out.weight = a * u.vertex_alpha_scale * mix(FADE_ALPHA, 1.0, f32(mask[ii]));
    return out;
}

@fragment
fn fs_points(in: POut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    // Same analytic AA rim as points.wgsl — coverage falloff, no MSAA.
    let aa = max(fwidth(d), 1e-4);
    let cov = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    if cov <= 0.0 {
        discard;
    }
    return vec4<f32>(in.weight * cov, 0.0, 0.0, 0.0);
}

// ---- edges -----------------------------------------------------------------

struct EOut {
    @builtin(position) clip_pos: vec4<f32>,
    // signed pixel offset from the segment centerline, for the 1px coverage falloff
    @location(0) edge_dist_px: f32,
    @location(1) @interpolate(flat) weight: f32,
};

@vertex
fn vs_edges(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> EOut {
    let edge = edges[ii];
    let clip_a = u.view_proj * vec4<f32>(positions[edge.x], 0.0, 1.0);
    let clip_b = u.view_proj * vec4<f32>(positions[edge.y], 0.0, 1.0);

    let half_vp = 0.5 * u.viewport_size;
    let px_a = clip_a.xy * half_vp;
    let px_b = clip_b.xy * half_vp;
    let seg = px_b - px_a;
    let len = length(seg);
    let dir = select(vec2<f32>(0.0, 0.0), seg / len, len > 1e-6);
    let perp = vec2<f32>(-dir.y, dir.x);
    let half_w = 0.5 * max(u.line_width_px * u.edge_width_scale, 1.0);
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);
    let offset_ndc = (perp * half_w * side) / max(half_vp, vec2<f32>(1.0, 1.0));

    let on_a = vi < 2u;
    let vid = select(edge.y, edge.x, on_a);
    let clip = select(clip_b, clip_a, on_a);

    var out: EOut;
    out.clip_pos = vec4<f32>(clip.xy + offset_ndc, clip.zw);
    out.edge_dist_px = side * half_w;
    // Per-edge weight = global edge-alpha scale × (weight-by-alpha if engaged) × file
    // alpha (RGB ignored) × endpoint fade.
    var weight_fade: f32 = 1.0;
    if u.edge_alpha_source == EDGE_ALPHA_BY_WEIGHT {
        weight_fade = edge_alphas[ii];
    }
    let file_a = unpack4x8unorm(edge_colors[ii]).a;
    out.weight = u.edge_alpha_scale * weight_fade * file_a * mix(FADE_ALPHA, 1.0, f32(mask[vid]));
    return out;
}

@fragment
fn fs_edges(in: EOut) -> @location(0) vec4<f32> {
    // 1px coverage falloff across the stroke instead of MSAA: fade the outer ~1px band.
    let aa = max(fwidth(in.edge_dist_px), 1e-4);
    let half_w = 0.5 * max(u.line_width_px * u.edge_width_scale, 1.0);
    let cov = 1.0 - smoothstep(half_w - aa, half_w, abs(in.edge_dist_px));
    if cov <= 0.0 {
        discard;
    }
    return vec4<f32>(in.weight * cov, 0.0, 0.0, 0.0);
}
