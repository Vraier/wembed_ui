// Instanced quad → antialiased disk. Each instance is one node's 2D world position;
// the vertex shader expands to a quad of size `point_radius` world units around it,
// and the fragment shader clips to a circle with a smoothstep edge.
//
// The `vertex_color_source` uniform picks the tint source:
//   VERTEX_COLOR_THEME (0)  — the theme's vertex color for every vertex.
//   VERTEX_COLOR_BUFFER (1) — per-vertex packed RGBA8 from the colors storage buffer
//                             (categories or a colors file, resolved CPU-side).

const VERTEX_COLOR_THEME: u32 = 0u;
const VERTEX_COLOR_BUFFER: u32 = 1u;

// Size-channel selectors — mirror `SIZE_*` on the Rust side and drive the size multiplier.
const SIZE_CONSTANT: u32 = 0u;
const SIZE_BY_WEIGHT: u32 = 1u;

struct Uniforms {
    view_proj: mat4x4<f32>,
    point_radius: f32,
    vertex_color_source: u32,
    vertex_size_source: u32,
    edge_alpha_source: u32,
    // Global multipliers driven by UI sliders. `vertex_size_scale` composes with the
    // per-vertex `sqrt(w/mean_w)` from size-by-weight; `vertex_alpha_scale` is a linear
    // opacity for vertices (0 = invisible); `edge_alpha_scale` is the same for edges and
    // is unused in this shader (declared for struct-layout parity with `lines.wgsl`).
    vertex_size_scale: f32,
    vertex_alpha_scale: f32,
    edge_alpha_scale: f32,
    // Edge-color fields — unused here, declared for struct-layout parity with `lines.wgsl`.
    edge_color_source: u32,
    edge_theme_color: vec4<f32>,
    edge_len_log_min: f32,
    edge_len_log_max: f32,
    // Strength of the depth shading (0 = off): vertex alpha is multiplied by
    // `1 - depth_fade * depth[i]`, fading vertices the 2D projection misrepresents.
    depth_fade: f32,
    // Edge fields — unused here, declared for struct-layout parity with `lines.wgsl`.
    line_width_px: f32,
    viewport_size: vec2<f32>,
    edge_width_scale: f32,
    _pad1: f32,
    // Vertex color of VERTEX_COLOR_THEME (theme base09), sRGB like the other colour inputs.
    vertex_theme_color: vec4<f32>,
};

// Uniforms live in @group(0), owned by Renderer. Per-vertex storage buffers live in
// @group(1), owned by Scene. The split lets Scene grow attributes without touching
// Renderer's bindings, and vice versa.
@group(0) @binding(0) var<uniform> u: Uniforms;
// Packed RGBA8 per vertex, indexed by `instance_index`. Little-endian byte order:
// byte0=R, byte1=G, byte2=B, byte3=A. Matches Rust's `u32::from_le_bytes([R,G,B,A])`,
// which is how the loader packs entries from the hex colors file.
@group(1) @binding(0) var<storage, read> colors: array<u32>;
// Highlight mask, one u32 per vertex. `1u` = bright (unmodified alpha), `0u` = faded
// (alpha multiplied by FADE_ALPHA). All-1s means "no highlight active." Kept as a
// separate SSBO from `colors` so search/hover don't have to rewrite color data.
@group(1) @binding(1) var<storage, read> mask: array<u32>;
// Per-vertex size multiplier, precomputed as `sqrt(w_i / mean_w)` in the embedder.
// Read only when `vertex_size_source == SIZE_BY_WEIGHT`; otherwise the value is ignored. The
// buffer is always present (fallback = all-1.0), so unconditional read is cheap.
@group(1) @binding(2) var<storage, read> values: array<f32>;
// Per-vertex depth-shade value in [0, 1] — the projection residual, quantile-normalized
// on the CPU. All-0 (no fade) at dimension 2; scaled by the `depth_fade` uniform.
@group(1) @binding(7) var<storage, read> depth: array<f32>;

// Alpha multiplier applied to faded (mask == 0) vertices. Aggressive so highlighted
// regions stand out against dense graphs — 5% survives cumulative overdraw better than
// a naïve 15%. Mirrors `highlight::FADE_ALPHA` on the Rust side (kept in lockstep by
// convention, not by build).
const FADE_ALPHA: f32 = 0.05;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
    // `1.0` for a highlighted vertex, `0.0` for a faded one. Interpolation across the
    // six quad corners is a no-op — all six share the same instance index.
    @location(2) highlight: f32,
};

// Two triangles covering [-1, 1]^2 in local quad space.
// Naga rejects dynamic indexing of a `const` array. `var<private>` allows it and the
// initializer is still folded to a constant table, so no runtime cost.
var<private> QUAD: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
    @location(0) instance_pos: vec2<f32>,
) -> VOut {
    let q = QUAD[vi];
    // Size channel: constant → base radius as-is, by-weight → base * sqrt(w/mean_w). The
    // pixel-clamp on the base still applies (via `point_radius` uniform); size-by-weight
    // just scales the resulting world radius. `point_size_scale` (UI slider) is a global
    // multiplier composed on top — a "how big is everything overall" knob.
    var size_mult: f32 = 1.0;
    if u.vertex_size_source == SIZE_BY_WEIGHT {
        size_mult = values[ii];
    }
    let world = instance_pos + q * u.point_radius * size_mult * u.vertex_size_scale;
    var out: VOut;
    out.clip_pos = u.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.uv = q;
    // Resolve tint at vertex granularity — all six quad corners share the same instance,
    // so an interpolated varying is wasted work but has zero visual effect at the disk
    // scale we operate at.
    var tint = u.vertex_theme_color;
    if u.vertex_color_source == VERTEX_COLOR_BUFFER {
        tint = unpack4x8unorm(colors[ii]);
    }
    out.tint = vec4<f32>(srgb_to_linear(tint.rgb), tint.a);
    out.highlight = f32(mask[ii]);
    // depth shading folds into the tint's alpha at vertex granularity, same rationale
    // as the tint itself
    out.tint.a = out.tint.a * (1.0 - u.depth_fade * depth[ii]);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    // Screen-space antialiased rim: `fwidth(d)` is how much d changes across one pixel,
    // so the feather band is ~1 pixel wide REGARDLESS of the disk's screen size. A
    // radius-relative band (the old `smoothstep(0.9, 1.0, d)`) blurred big vertices —
    // 10% of an 80-pixel hub is an 8-pixel fuzz. Floored so degenerate derivatives
    // (sub-pixel dots) can't produce a zero-width step.
    let aa = max(fwidth(d), 1e-4);
    let edge = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    if edge <= 0.0 {
        discard;
    }
    // Fade multiplier: 1.0 for highlighted verts, FADE_ALPHA for the rest.
    let fade = mix(FADE_ALPHA, 1.0, in.highlight);
    // Global vertex-alpha slider composes multiplicatively with everything else. `0.0`
    // makes vertices invisible without a pipeline gate.
    return vec4<f32>(in.tint.rgb, in.tint.a * edge * fade * u.vertex_alpha_scale);
}
