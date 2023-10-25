@group(0) @binding(4)
var point_shadow_textures: texture_depth_cube_array;
@group(0) @binding(5)
var point_shadow_textures_sampler: sampler_comparison;

@fragment
fn fragment() -> @location(0) vec4<f32> {
    let frag_ls = vec4<f32>(1., 1., 2., 1.).xyz;
    let a = textureSampleCompare(point_shadow_textures, point_shadow_textures_sampler, frag_ls, i32(1), 1.);

    return vec4<f32>(a, 1., 1., 1.);
}
