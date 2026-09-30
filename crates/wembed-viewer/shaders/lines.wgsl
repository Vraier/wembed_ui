// Edges: instanced non-indexed triangle-strip draw — each edge is a screen-space quad.
//
// Each edge is one instance of a 4-vertex triangle strip (one quad). The vertex shader
// consumes:
//   - `@builtin(vertex_index)` — 0..3. 0,1 sit at endpoint A offset ±half_width along the
//                                 screen-space perpendicular; 2,3 at endpoint B. The strip
//                                 (A-, A+, B-, B+) forms the two triangles of the quad.
//   - `@builtin(instance_index)` — the edge id, used to look up the endpoint pair from the
//                                  `edges` SSBO and the precomputed alpha from `edge_alphas`.
//
// WHY a quad and not `PrimitiveTopology::LineList`: WebGPU hardware lines are always exactly
// 1 device pixel, so a line's relative thickness is `1/render_height` — 4K/8K exports come
// out with half/quarter the stroke width of the ~1080p interactive window. Expanding to a
// pixel-sized quad lets `line_width_px` scale with render height (mirrors what points do via
// `scaled_sizing`), so the same config reads identically at any resolution. At 1080p the
// quad is 1px wide, visually indistinguishable from the old hardware line.
//
// This shape (instanced + SSBO reads) exists so per-edge data (endpoints, alpha, per-edge
// color) can be a `@interpolate(flat)` varying with **no interpolation** across the segment,
// which indexed drawing over the shared position buffer can't express (vertex attributes are
// fetched by the index buffer, so different edges hitting the same graph vertex all see the
// same value). The `vertex` color source is the deliberate exception: it emits each
// endpoint's own color as a *linearly-interpolated* varying. Endpoint-A vertices (0,1) carry
// endpoint-A values and endpoint-B vertices (2,3) carry endpoint-B values, so interpolation
// along the segment length behaves exactly as it did with the two-vertex line.
//
// Highlight interaction: look up each endpoint's mask bit, emit as a varying, fragment sees
// linearly-interpolated highlight from bright→faded across the segment when only one endpoint
// is highlighted. Approximate but visually pleasant.

// Edge-alpha channel selectors — mirror `EDGE_ALPHA_*` on the Rust side.
const EDGE_ALPHA_CONSTANT: u32 = 0u;
const EDGE_ALPHA_BY_WEIGHT: u32 = 1u;

// Edge-color source selectors — mirror `EDGE_COLOR_*` in `cli::EdgeColorSource::shader_value`.
const EDGE_COLOR_THEME: u32 = 0u;   // fixed theme color (u.edge_theme_color)
const EDGE_COLOR_VERTEX: u32 = 1u;  // interpolate endpoint vertex colors (rgb only)
const EDGE_COLOR_FILE: u32 = 2u;    // per-edge baked color (rgb + alpha multiplier)
const EDGE_COLOR_LENGTH: u32 = 3u;  // computed stress: |dist| / (w_a*w_b)^(1/2) on a diverging map

// Diverging colormap endpoints for the `length` source, in sRGB (mixed in sRGB, decoded
// with the other edge colours): compressed→cool, ideal→neutral, stretched→warm.
const EDGE_LEN_COOL: vec3<f32> = vec3<f32>(0.20, 0.42, 0.90);
const EDGE_LEN_NEUTRAL: vec3<f32> = vec3<f32>(0.62, 0.62, 0.66);
const EDGE_LEN_WARM: vec3<f32> = vec3<f32>(0.92, 0.32, 0.16);

struct Uniforms {
    view_proj: mat4x4<f32>,
    point_radius: f32,
    vertex_color_source: u32,
    vertex_size_source: u32,
    edge_alpha_source: u32,
    // Global multipliers from UI sliders. `edge_alpha_scale` replaces the historical
    // hardcoded `0.35` edge base alpha; `vertex_size_scale` and `vertex_alpha_scale` are
    // unused in this shader but declared for struct-layout parity with `points.wgsl`.
    vertex_size_scale: f32,
    vertex_alpha_scale: f32,
    edge_alpha_scale: f32,
    // Edge-color source selector (EDGE_COLOR_*). Replaces the old struct-tail padding.
    edge_color_source: u32,
    // Theme edge color for EDGE_COLOR_THEME (rgb; a unused). Normalized [0,1] sRGB bytes,
    // decoded like the vertex colors.
    edge_theme_color: vec4<f32>,
    // ln(r) clamp bounds for EDGE_COLOR_LENGTH: log_min = ln(range.lo) (<0), log_max =
    // ln(range.hi) (>0). r=1 (ln 0) is the neutral midpoint.
    edge_len_log_min: f32,
    edge_len_log_max: f32,
    // Depth-shading strength — unused here (edges keep full alpha), declared for
    // struct-layout parity with `points.wgsl`.
    depth_fade: f32,
    // Edge stroke width in device pixels. Points ignore it. `1.0` at REFERENCE_HEIGHT
    // (1080p) reproduces the old 1-device-pixel hardware line; scales with render height
    // so 4K/8K exports keep the same relative thickness.
    line_width_px: f32,
    // Render-target size in device pixels (width, height). Used to convert the segment's
    // clip-space endpoints to pixel space for the perpendicular offset. Points ignore it.
    viewport_size: vec2<f32>,
    edge_width_scale: f32,
    _pad1: f32,
    // Unused here, declared for struct-layout parity with `points.wgsl`.
    vertex_theme_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Packed RGBA8 per graph vertex — same buffer points reads. Used by EDGE_COLOR_VERTEX.
@group(1) @binding(0) var<storage, read> colors: array<u32>;
@group(1) @binding(1) var<storage, read> mask: array<u32>;
// Positions of every graph vertex — indexed by graph vertex id, looked up via the edges
// SSBO. Same buffer that points uses as a VBO.
@group(1) @binding(3) var<storage, read> positions: array<vec2<f32>>;
// One entry per edge — `[src_vid, dst_vid]`. Indexed by `@builtin(instance_index)`.
@group(1) @binding(4) var<storage, read> edges: array<vec2<u32>>;
// Precomputed per-edge alpha multiplier `1/(w_a * w_b)`. Same indexing as `edges`.
@group(1) @binding(5) var<storage, read> edge_alphas: array<f32>;
// Per-edge baked color (packed RGBA8) for EDGE_COLOR_FILE. Same indexing as `edges`.
@group(1) @binding(6) var<storage, read> edge_colors: array<u32>;
// Per-edge stress ratio `|pos_a − pos_b| / (w_a·w_b)^(1/dim)` for EDGE_COLOR_LENGTH,
// computed CPU-side over the FULL d-dim positions (the projected 2D distance here would
// underestimate, and the weight exponent depends on the embedding dimension).
@group(1) @binding(8) var<storage, read> edge_lengths: array<f32>;

const FADE_ALPHA: f32 = 0.05;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    // Edge rgb, linear light. Interpolated so EDGE_COLOR_VERTEX blends the two endpoint
    // colors; the other sources emit the same value at both endpoints (interpolation no-op).
    @location(0) rgb: vec3<f32>,
    // `1.0` for a highlighted endpoint, `0.0` for a faded one. Linearly interpolated.
    @location(1) highlight: f32,
    // Per-edge alpha multiplier (weight-driven). Flat: both endpoints emit the same value.
    @location(2) @interpolate(flat) edge_alpha: f32,
    // File-mode alpha multiplier (the 8-digit hex alpha). `1.0` for every other source.
    @location(3) @interpolate(flat) file_alpha: f32,
};

// Diverging colormap for the `length` source. `r` is the stress ratio (drawn length /
// ideal length); r<1 compressed (cool), r=1 ideal (neutral), r>1 stretched (warm). log(r)
// is clamped to the configured [log_min, log_max] window, symmetric about 0 by default.
fn edge_length_color(r: f32) -> vec3<f32> {
    let s = log(max(r, 1e-6));
    if s >= 0.0 {
        let t = clamp(s / max(u.edge_len_log_max, 1e-6), 0.0, 1.0);
        return mix(EDGE_LEN_NEUTRAL, EDGE_LEN_WARM, t);
    }
    // s<0 and log_min<0 → the ratio is a positive 0..1 fraction toward the cool end.
    let t = clamp(s / min(u.edge_len_log_min, -1e-6), 0.0, 1.0);
    return mix(EDGE_LEN_NEUTRAL, EDGE_LEN_COOL, t);
}

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,        // 0..3: quad corner (0,1=endpoint A; 2,3=endpoint B)
    @builtin(instance_index) ii: u32,      // edge id
) -> VOut {
    let edge = edges[ii];
    // Both endpoints in clip space (camera is 2D orthographic, w==1). Both are needed on
    // every vertex so the pixel-space perpendicular of the whole segment can be built.
    let clip_a = u.view_proj * vec4<f32>(positions[edge.x], 0.0, 1.0);
    let clip_b = u.view_proj * vec4<f32>(positions[edge.y], 0.0, 1.0);

    // Screen-space direction A→B in pixels (NDC → pixels: ×0.5×viewport per axis). The
    // 0.5 factors cancel in the normalize, but keep them for a correct-magnitude fallback.
    let half_vp = 0.5 * u.viewport_size;
    let px_a = clip_a.xy * half_vp;
    let px_b = clip_b.xy * half_vp;
    let seg = px_b - px_a;
    let len = length(seg);
    // Degenerate (coincident endpoints): emit a zero-area quad by using a zero perpendicular.
    let dir = select(vec2<f32>(0.0, 0.0), seg / len, len > 1e-6);
    let perp = vec2<f32>(-dir.y, dir.x);
    // Offset ±half_width in pixel space, back to NDC (÷ half_vp).
    // floored at 1px: thinner quads alias into dashes
    let half_w = 0.5 * max(u.line_width_px * u.edge_width_scale, 1.0);
    let side = select(-1.0, 1.0, (vi & 1u) == 1u);   // 0,2 → −side; 1,3 → +side
    let offset_ndc = (perp * half_w * side) / max(half_vp, vec2<f32>(1.0, 1.0));

    // vi<2 → endpoint A, else endpoint B. Endpoint-A vertices carry A's varyings, B's carry
    // B's — so the fragment sees the same linear blend the two-vertex line produced.
    let on_a = vi < 2u;
    let vid = select(edge.y, edge.x, on_a);
    let clip = select(clip_b, clip_a, on_a);

    var out: VOut;
    out.clip_pos = vec4<f32>(clip.xy + offset_ndc, clip.zw);
    out.highlight = f32(mask[vid]);
    out.edge_alpha = edge_alphas[ii];

    var rgb = u.edge_theme_color.rgb;
    var file_alpha = 1.0;
    if u.edge_color_source == EDGE_COLOR_VERTEX {
        rgb = unpack4x8unorm(colors[vid]).rgb;
    } else if u.edge_color_source == EDGE_COLOR_FILE {
        let c = unpack4x8unorm(edge_colors[ii]);
        rgb = c.rgb;
        file_alpha = c.a;
    } else if u.edge_color_source == EDGE_COLOR_LENGTH {
        rgb = edge_length_color(edge_lengths[ii]);
    }
    out.rgb = srgb_to_linear(rgb);
    out.file_alpha = file_alpha;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let fade = mix(FADE_ALPHA, 1.0, in.highlight);
    // Per-edge weight multiplier, gated by the channel selector so `constant` skips it.
    var weight_fade: f32 = 1.0;
    if u.edge_alpha_source == EDGE_ALPHA_BY_WEIGHT {
        weight_fade = in.edge_alpha;
    }
    // `file_alpha` is 1.0 for every source except EDGE_COLOR_FILE, so the 8-digit hex alpha
    // multiplies in without gating (a junction edge authored `...00` disappears here).
    return vec4<f32>(in.rgb, u.edge_alpha_scale * fade * weight_fade * in.file_alpha);
}
