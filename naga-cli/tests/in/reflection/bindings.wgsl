// Immediate data, binding arrays, and dual-source blending.
enable dual_source_blending;
enable wgpu_binding_array;

struct Immediates {
    tint: vec4<f32>,
    index: u32,
}

var<immediate> immediates: Immediates;

@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 4>;
@group(0) @binding(1) var samplers: binding_array<sampler>;

struct BlendOutput {
    @location(0) @blend_src(0) color: vec4<f32>,
    @location(0) @blend_src(1) blend: vec4<f32>,
}

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> BlendOutput {
    let c = textureSample(textures[immediates.index], samplers[immediates.index], uv);
    return BlendOutput(c * immediates.tint, c);
}
