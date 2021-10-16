[[group(0), binding(0)]]
var image_mipmapped_src: texture_2d<u32>;
[[group(0), binding(3)]]
var image_multisampled_src: texture_multisampled_2d<u32>;
[[group(0), binding(4)]]
var image_depth_multisampled_src: texture_depth_multisampled_2d;
[[group(0), binding(1)]]
var image_storage_src: texture_storage_2d<rgba8uint,read>;
[[group(0), binding(5)]]
var image_array_src: texture_2d_array<u32>;
[[group(0), binding(6)]]
var image_dup_src: texture_storage_1d<r32uint,read>;
[[group(0), binding(2)]]
var image_dst: texture_storage_1d<r32uint,write>;
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
    let dim: vec2<i32> = textureDimensions(image_storage_src);
    let itc: vec2<i32> = ((dim * vec2<i32>(local_id.xy)) % vec2<i32>(10, 20));
    let value1: vec4<u32> = textureLoad(image_mipmapped_src, itc, i32(local_id.z));
    let value2: vec4<u32> = textureLoad(image_multisampled_src, itc, i32(local_id.z));
    let value4: vec4<u32> = textureLoad(image_storage_src, itc);
    let value5: vec4<u32> = textureLoad(image_array_src, itc, i32(local_id.z), (i32(local_id.z) + 1));
    textureStore(image_dst, itc.x, (((value1 + value2) + value4) + value5));
    return;
}

[[stage(compute), workgroup_size(16, 1, 1)]]
fn depth_load([[builtin(local_invocation_id)]] local_id1: vec3<u32>) {
    let dim1: vec2<i32> = textureDimensions(image_storage_src);
    let itc1: vec2<i32> = ((dim1 * vec2<i32>(local_id1.xy)) % vec2<i32>(10, 20));
    let val: f32 = textureLoad(image_depth_multisampled_src, itc1, i32(local_id1.z));
    textureStore(image_dst, itc1.x, vec4<u32>(u32(val)));
    return;
}

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    let dim_1d: i32 = textureDimensions(image_1d);
    let dim_2d: vec2<i32> = textureDimensions(image_2d);
    let dim_2d_lod: vec2<i32> = textureDimensions(image_2d, 1);
    let dim_2d_array: vec2<i32> = textureDimensions(image_2d_array);
    let dim_2d_array_lod: vec2<i32> = textureDimensions(image_2d_array, 1);
    let dim_cube: vec2<i32> = textureDimensions(image_cube);
    let dim_cube_lod: vec2<i32> = textureDimensions(image_cube, 1);
    let dim_cube_array: vec2<i32> = textureDimensions(image_cube_array);
    let dim_cube_array_lod: vec2<i32> = textureDimensions(image_cube_array, 1);
    let dim_3d: vec3<i32> = textureDimensions(image_3d);
    let dim_3d_lod: vec3<i32> = textureDimensions(image_3d, 1);
    let sum: i32 = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    return vec4<f32>(f32(sum));
}

[[stage(vertex)]]
fn levels_queries() -> [[builtin(position)]] vec4<f32> {
    let num_levels_2d: i32 = textureNumLevels(image_2d);
    let num_levels_2d_array: i32 = textureNumLevels(image_2d_array);
    let num_layers_2d: i32 = textureNumLayers(image_2d_array);
    let num_levels_cube: i32 = textureNumLevels(image_cube);
    let num_levels_cube_array: i32 = textureNumLevels(image_cube_array);
    let num_layers_cube: i32 = textureNumLayers(image_cube_array);
    let num_levels_3d: i32 = textureNumLevels(image_3d);
    let num_samples_aa: i32 = textureNumSamples(image_aa);
    let sum1: i32 = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return vec4<f32>(f32(sum1));
}

[[stage(fragment)]]
fn sample() -> [[location(0)]] vec4<f32> {
    let tc: vec2<f32> = vec2<f32>(0.5);
    let s1d: vec4<f32> = textureSample(image_1d, sampler_reg, tc.x);
    let s2d: vec4<f32> = textureSample(image_2d, sampler_reg, tc);
    let s2d_offset: vec4<f32> = textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let s2d_level: vec4<f32> = textureSampleLevel(image_2d, sampler_reg, tc, 2.299999952316284);
    let s2d_level_offset: vec4<f32> = textureSampleLevel(image_2d, sampler_reg, tc, 2.299999952316284, vec2<i32>(3, 1));
    return ((((s1d + s2d) + s2d_offset) + s2d_level) + s2d_level_offset);
}

[[stage(fragment)]]
fn sample_comparison() -> [[location(0)]] f32 {
    let tc1: vec2<f32> = vec2<f32>(0.5);
    let s2d_depth: f32 = textureSampleCompare(image_2d_depth, sampler_cmp, tc1, 0.5);
    let s2d_depth_level: f32 = textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc1, 0.5);
    return (s2d_depth + s2d_depth_level);
}
