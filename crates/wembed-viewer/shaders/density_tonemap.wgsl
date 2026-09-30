// Density tone-map: read the blurred R16Float density, normalize to t∈[0,1] via the log
// curve anchored at (knee, white), and look up color in the fused LUT (curve already baked
// into the ramp's t-axis, so this pass only needs the t→texel sample). Writes to the sRGB
// LDR target. A fullscreen-triangle pass; runs AFTER the scene would (leaving room for a
// future crisp-linework overlay between here and the text pass) and BEFORE the text pass.
//
// The curve here MUST match the LUT builder's `curve_t` on the Rust side — the LUT is built
// over uniform t, so we map density→t identically, then sample the LUT at t.

struct ToneUniforms {
    knee: f32,        // density anchor where t starts rising (default 0: no floor plateau)
    white: f32,       // density anchor mapped to t = 1
    // 1 / log1p((white - knee)/scale) — normalizer so density=white → t=1. Precomputed CPU-side.
    inv_log_span: f32,
    scale: f32,       // log1p denominator scale (knee-relative); see curve_t on the Rust side
};

@group(0) @binding(0) var<uniform> u: ToneUniforms;
@group(0) @binding(1) var density_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_tex: texture_2d<f32>;   // 512×1 RGBA8 (sRGB), curve+ramp fused
@group(0) @binding(3) var lut_sampler: sampler;       // linear, clamp

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// density → t∈[0,1] on the log curve. Keep in lockstep with `DensityLut::curve_t`.
fn curve_t(d: f32) -> f32 {
    let x = max(d - u.knee, 0.0);
    let t = log(1.0 + x / u.scale) * u.inv_log_span;
    return clamp(t, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let d = textureLoad(density_tex, vec2<i32>(frag.xy), 0).r;
    // No unlit special case: ramp[0] IS the canvas color by definition. A branch on d==0
    // renders the separable blur's SQUARE support as a hard silhouette (any nonzero tail
    // → floor color) and needs a bg uniform that must round-trip sRGB exactly like the LUT.
    let t = curve_t(d);
    // Sample at texel centers: LUT is 512 wide, so map t∈[0,1] into [0.5/512, 511.5/512].
    let uv = vec2<f32>((t * 511.0 + 0.5) / 512.0, 0.5);
    let c = textureSampleLevel(lut_tex, lut_sampler, uv, 0.0);
    return vec4<f32>(c.rgb, 1.0);
}
