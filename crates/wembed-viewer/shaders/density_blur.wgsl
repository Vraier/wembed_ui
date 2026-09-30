// Separable gaussian blur over the density accumulation texture. One axis per pass
// (ping-pong): the caller runs horizontal then vertical. Weights are a normalized gaussian
// with sigma in PIXELS; kernel radius = ceil(3σ) uploaded as `radius`. σ=0 is bypassed by
// the caller (both passes skipped) so this shader always has radius >= 1.
//
// Fullscreen triangle (3 verts, no vertex buffer). Reads exact texels (no filtering needed
// — R16Float sampling stays nearest), sums cov·weight, writes the smoothed scalar.

struct BlurUniforms {
    // pixel step for this pass: (1,0) horizontal, (0,1) vertical
    axis: vec2<f32>,
    radius: i32,
    inv_two_sigma_sq: f32, // 1 / (2σ²) for the gaussian falloff
};

@group(0) @binding(0) var<uniform> u: BlurUniforms;
@group(0) @binding(1) var src: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // Oversized triangle covering the viewport.
    var out: VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.clip_pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) f32 {
    let center = vec2<i32>(frag.xy);
    let dims = vec2<i32>(textureDimensions(src, 0));
    let step = vec2<i32>(u.axis);
    var acc: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var i: i32 = -u.radius; i <= u.radius; i = i + 1) {
        let w = exp(-f32(i * i) * u.inv_two_sigma_sq);
        // clamp-to-edge so the border doesn't darken the blur
        let p = clamp(center + step * i, vec2<i32>(0), dims - vec2<i32>(1));
        acc = acc + w * textureLoad(src, p, 0).r;
        wsum = wsum + w;
    }
    return acc / wsum;
}
