enable resource_table;

@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var samp_cmp: sampler_comparison;
@group(0) @binding(2) var<uniform> uniform_index: u32;

const FIXED_INDEX: u32 = 7u;

@fragment
fn fs_main(
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) dynamic_index: u32,
) -> @location(0) vec4<f32> {
    // Uniform (constant) index.
    let tex_const = getResource<texture_2d<f32>>(FIXED_INDEX);
    let color_const = textureSample(tex_const, samp, pos.xy);

    // Uniform (buffer-derived) index.
    let tex_uniform = getResource<texture_2d<f32>>(uniform_index);
    let color_uniform = textureSample(tex_uniform, samp, pos.xy);

    // Non-uniform index.
    let tex_dynamic = getResource<texture_2d<f32>>(dynamic_index);
    let color_dynamic = textureSample(tex_dynamic, samp, pos.xy);

    // Depth texture with comparison sampling.
    let depth = getResource<texture_depth_2d>(dynamic_index + 1u);
    let shadow = textureSampleCompare(depth, samp_cmp, pos.xy, 0.5);

    // Arrayed texture via textureLoad.
    let tex_array = getResource<texture_2d_array<u32>>(uniform_index);
    let texel_array = textureLoad(tex_array, vec2<i32>(0, 0), 3, 0);

    // Multisampled texture via textureLoad.
    let tex_ms = getResource<texture_multisampled_2d<f32>>(0u);
    let texel_ms = textureLoad(tex_ms, vec2<i32>(0, 0), 2);

    return color_const + color_uniform + color_dynamic
        + vec4(shadow) + vec4<f32>(texel_array) + texel_ms;
}
