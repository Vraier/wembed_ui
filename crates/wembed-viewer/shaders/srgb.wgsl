// Prepended to points.wgsl and lines.wgsl (scene_renderer.rs). Every colour input — hex
// files, theme keys, shader constants — is authored in sRGB, and the *Srgb render target
// encodes on write, so colours must reach the blender as linear light or they get encoded
// twice (renders paler than authored).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}
