[[group(0), binding(1)]]
var image_src: [[access(read)]] texture_storage_2d<rgba8uint>;
[[group(0), binding(2)]]
var image_dst: [[access(write)]] texture_storage_1d<r32uint>;
[[group(0), binding(0)]]
var image_1d: texture_1d<f32>;
[[group(0), binding(1)]]
var image_2d: texture_2d<f32>;
[[group(0), binding(2)]]
var image_2d_array: texture_2d_array<f32>;
[[group(0), binding(3)]]
var image_cube: texture_cube<f32>;
[[group(0), binding(4)]]
var image_cube_array: texture_cube_array<f32>;
[[group(0), binding(5)]]
var image_3d: texture_3d<f32>;
[[group(0), binding(6)]]
var image_aa: texture_multisampled_2d<f32>;
[[group(1), binding(0)]]
var sampler_reg: sampler;
[[group(1), binding(1)]]
var sampler_cmp: sampler_comparison;
[[group(1), binding(2)]]
var image_2d_depth: texture_depth_2d;

[[stage(compute), workgroup_size(16, 1, 1)]]
fn main([[builtin(local_invocation_id)]] local_id: vec3<u32>) {
    let dim: vec2<i32> = textureDimensions(image_src);
    let itc: vec2<i32> = ((dim * vec2<i32>(local_id.xy)) % vec2<i32>(10, 20));
    let value: vec4<u32> = textureLoad(image_src, itc);
    textureStore(image_dst, itc.x, value);
    return;
}

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    let dim_1d: i32 = textureDimensions(image_1d);
    let dim_2d: vec2<i32> = textureDimensions(image_2d);
    let num_levels_2d: i32 = textureNumLevels(image_2d);
    let dim_2d_lod: vec2<i32> = textureDimensions(image_2d, 1);
    let dim_2d_array: vec2<i32> = textureDimensions(image_2d_array);
    let num_levels_2d_array: i32 = textureNumLevels(image_2d_array);
    let dim_2d_array_lod: vec2<i32> = textureDimensions(image_2d_array, 1);
    let num_layers_2d: i32 = textureNumLayers(image_2d_array);
    let dim_cube: vec2<i32> = textureDimensions(image_cube);
    let num_levels_cube: i32 = textureNumLevels(image_cube);
    let dim_cube_lod: vec2<i32> = textureDimensions(image_cube, 1);
    let dim_cube_array: vec2<i32> = textureDimensions(image_cube_array);
    let num_levels_cube_array: i32 = textureNumLevels(image_cube_array);
    let dim_cube_array_lod: vec2<i32> = textureDimensions(image_cube_array, 1);
    let num_layers_cube: i32 = textureNumLayers(image_cube_array);
    let dim_3d: vec3<i32> = textureDimensions(image_3d);
    let num_levels_3d: i32 = textureNumLevels(image_3d);
    let dim_3d_lod: vec3<i32> = textureDimensions(image_3d, 1);
    let num_samples_aa: i32 = textureNumSamples(image_aa);
    let sum: i32 = ((((((((((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + num_layers_2d) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + num_layers_cube) + dim_3d.z) + dim_3d_lod.z) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return vec4<f32>(f32(sum));
}

[[stage(fragment)]]
fn sample() -> [[location(0)]] vec4<f32> {
    let tc: vec2<f32> = vec2<f32>(0.5);
    let s2d: vec4<f32> = textureSample(image_2d, sampler_reg, tc);
    let s2d_offset: vec4<f32> = textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let s2d_level: vec4<f32> = textureSampleLevel(image_2d, sampler_reg, tc, 2.3);
    let s2d_level_offset: vec4<f32> = textureSampleLevel(image_2d, sampler_reg, tc, 2.3, vec2<i32>(3, 1));
    return (((s2d + s2d_offset) + s2d_level) + s2d_level_offset);
}

[[stage(fragment)]]
fn sample_comparison() -> [[location(0)]] f32 {
    let tc: vec2<f32> = vec2<f32>(0.5);
    let s2d_depth: f32 = textureSampleCompare(image_2d_depth, sampler_cmp, tc, 0.5);
    let s2d_depth_level: f32 = textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc, 0.5);
    return (s2d_depth + s2d_depth_level);
}
