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
[[group(0), binding(7)]]
var image_1d_src: texture_1d<u32>;
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
    let dim = textureDimensions(image_storage_src);
    let itc = ((dim * vec2<i32>(local_id.xy)) % vec2<i32>(10, 20));
    let value1_ = textureLoad(image_mipmapped_src, itc, i32(local_id.z));
    let value2_ = textureLoad(image_multisampled_src, itc, i32(local_id.z));
    let value4_ = textureLoad(image_storage_src, itc);
    let value5_ = textureLoad(image_array_src, itc, i32(local_id.z), (i32(local_id.z) + 1));
    let value6_ = textureLoad(image_1d_src, i32(local_id.x), i32(local_id.z));
    textureStore(image_dst, itc.x, ((((value1_ + value2_) + value4_) + value5_) + value6_));
    return;
}

[[stage(compute), workgroup_size(16, 1, 1)]]
fn depth_load([[builtin(local_invocation_id)]] local_id_1: vec3<u32>) {
    let dim_1 = textureDimensions(image_storage_src);
    let itc_1 = ((dim_1 * vec2<i32>(local_id_1.xy)) % vec2<i32>(10, 20));
    let val = textureLoad(image_depth_multisampled_src, itc_1, i32(local_id_1.z));
    textureStore(image_dst, itc_1.x, vec4<u32>(u32(val)));
    return;
}

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    let dim_1d = textureDimensions(image_1d);
    let dim_1d_lod = textureDimensions(image_1d, i32(dim_1d));
    let dim_2d = textureDimensions(image_2d);
    let dim_2d_lod = textureDimensions(image_2d, 1);
    let dim_2d_array = textureDimensions(image_2d_array);
    let dim_2d_array_lod = textureDimensions(image_2d_array, 1);
    let dim_cube = textureDimensions(image_cube);
    let dim_cube_lod = textureDimensions(image_cube, 1);
    let dim_cube_array = textureDimensions(image_cube_array);
    let dim_cube_array_lod = textureDimensions(image_cube_array, 1);
    let dim_3d = textureDimensions(image_3d);
    let dim_3d_lod = textureDimensions(image_3d, 1);
    let sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    return vec4<f32>(f32(sum));
}

[[stage(vertex)]]
fn levels_queries() -> [[builtin(position)]] vec4<f32> {
    let num_levels_2d = textureNumLevels(image_2d);
    let num_levels_2d_array = textureNumLevels(image_2d_array);
    let num_layers_2d = textureNumLayers(image_2d_array);
    let num_levels_cube = textureNumLevels(image_cube);
    let num_levels_cube_array = textureNumLevels(image_cube_array);
    let num_layers_cube = textureNumLayers(image_cube_array);
    let num_levels_3d = textureNumLevels(image_3d);
    let num_samples_aa = textureNumSamples(image_aa);
    let sum_1 = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return vec4<f32>(f32(sum_1));
}

[[stage(fragment)]]
fn sample() -> [[location(0)]] vec4<f32> {
    let tc = vec2<f32>(0.5);
    let s1d = textureSample(image_1d, sampler_reg, tc.x);
    let s2d = textureSample(image_2d, sampler_reg, tc);
    let s2d_offset = textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let s2d_level = textureSampleLevel(image_2d, sampler_reg, tc, 2.299999952316284);
    let s2d_level_offset = textureSampleLevel(image_2d, sampler_reg, tc, 2.299999952316284, vec2<i32>(3, 1));
    return ((((s1d + s2d) + s2d_offset) + s2d_level) + s2d_level_offset);
}

[[stage(fragment)]]
fn sample_comparison() -> [[location(0)]] f32 {
    let tc_1 = vec2<f32>(0.5);
    let s2d_depth = textureSampleCompare(image_2d_depth, sampler_cmp, tc_1, 0.5);
    let s2d_depth_level = textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc_1, 0.5);
    return (s2d_depth + s2d_depth_level);
}

[[stage(fragment)]]
fn gather() -> [[location(0)]] vec4<f32> {
    let tc_2 = vec2<f32>(0.5);
    let s2d_1 = textureGather(1, image_2d, sampler_reg, tc_2);
    let s2d_offset_1 = textureGather(3, image_2d, sampler_reg, tc_2, vec2<i32>(3, 1));
    let s2d_depth_1 = textureGatherCompare(image_2d_depth, sampler_cmp, tc_2, 0.5);
    let s2d_depth_offset = textureGatherCompare(image_2d_depth, sampler_cmp, tc_2, 0.5, vec2<i32>(3, 1));
    return (((s2d_1 + s2d_offset_1) + s2d_depth_1) + s2d_depth_offset);
}

[[stage(fragment)]]
fn depth_no_comparison() -> [[location(0)]] vec4<f32> {
    let tc_3 = vec2<f32>(0.5);
    let s2d_2 = textureSample(image_2d_depth, sampler_reg, tc_3);
    let s2d_gather = textureGather(image_2d_depth, sampler_reg, tc_3);
    return (vec4<f32>(s2d_2) + s2d_gather);
}
