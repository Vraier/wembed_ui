// Textured quad for the CPU-rasterized title bitmap. Positioned directly in NDC via a
// rect uniform — no view_proj involvement, the title is screen-anchored. Drawn by both
// the window pass and the offscreen export pass so preview and PNG share identical
// compositing (blend happens in linear space on the sRGB target either way).

struct TextUniforms {
    // NDC rect: x0, y0, x1, y1 (y up). Written before every pass by the caller.
    rect: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: TextUniforms;
@group(0) @binding(1) var glyph_tex: texture_2d<f32>;
@group(0) @binding(2) var glyph_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Two triangles from a 6-vertex draw. `t` selects the quad corner in [0,1]^2.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let t = corners[vi];
    let x = mix(u.rect.x, u.rect.z, t.x);
    let y = mix(u.rect.y, u.rect.w, t.y);
    var out: VsOut;
    out.clip_pos = vec4<f32>(x, y, 0.0, 1.0);
    // NDC y grows up but the bitmap's row 0 is the top — flip v.
    out.uv = vec2<f32>(t.x, 1.0 - t.y);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(glyph_tex, glyph_sampler, in.uv);
}
