[[group(0), binding(0)]]
var image_mipmapped_src: texture_2d<u32>;
[[group(0), binding(3)]]
var image_multisampled_src: texture_multisampled_2d<u32>;
[[group(0), binding(4)]]
var image_depth_multisampled_src: texture_depth_multisampled_2d;
[[group(0), binding(1)]]
var image_storage_src: texture_storage_2d<rgba8uint, read>;
[[group(0), binding(5)]]
var image_array_src: texture_2d_array<u32>;
[[group(0), binding(6)]]
var image_dup_src: texture_storage_1d<r32uint,read>; // for #1307
[[group(0), binding(2)]]
var image_dst: texture_storage_1d<r32uint,write>;

[[stage(compute), workgroup_size(16)]]
fn main(
    [[builtin(local_invocation_id)]] local_id: vec3<u32>,
    //TODO: https://github.com/gpuweb/gpuweb/issues/1590
    //[[builtin(workgroup_size)]] wg_size: vec3<u32>
) {
    let dim = textureDimensions(image_storage_src);
    let itc = dim * vec2<i32>(local_id.xy) % vec2<i32>(10, 20);
    let value1 = textureLoad(image_mipmapped_src, itc, i32(local_id.z));
    let value2 = textureLoad(image_multisampled_src, itc, i32(local_id.z));
    let value3 = textureLoad(image_depth_multisampled_src, itc, i32(local_id.z));
    let value4 = textureLoad(image_storage_src, itc);
    let value5 = textureLoad(image_array_src, itc, i32(local_id.z), i32(local_id.z) + 1);
    textureStore(image_dst, itc.x, value1 + value2 + u32(value3) + value4 + value5);
}

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

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    let dim_1d = textureDimensions(image_1d);
    let dim_2d = textureDimensions(image_2d);
    let num_levels_2d = textureNumLevels(image_2d);
    let dim_2d_lod = textureDimensions(image_2d, 1);
    let dim_2d_array = textureDimensions(image_2d_array);
    let num_levels_2d_array = textureNumLevels(image_2d_array);
    let dim_2d_array_lod = textureDimensions(image_2d_array, 1);
    let num_layers_2d = textureNumLayers(image_2d_array);
    let dim_cube = textureDimensions(image_cube);
    let num_levels_cube = textureNumLevels(image_cube);
    let dim_cube_lod = textureDimensions(image_cube, 1);
    let dim_cube_array = textureDimensions(image_cube_array);
    let num_levels_cube_array = textureNumLevels(image_cube_array);
    let dim_cube_array_lod = textureDimensions(image_cube_array, 1);
    let num_layers_cube = textureNumLayers(image_cube_array);
    let dim_3d = textureDimensions(image_3d);
    let num_levels_3d = textureNumLevels(image_3d);
    let dim_3d_lod = textureDimensions(image_3d, 1);
    let num_samples_aa = textureNumSamples(image_aa);

    let sum = dim_1d + dim_2d.y + dim_2d_lod.y + dim_2d_array.y + dim_2d_array_lod.y +
        num_layers_2d + dim_cube.y + dim_cube_lod.y + dim_cube_array.y + dim_cube_array_lod.y +
        num_layers_cube + dim_3d.z + dim_3d_lod.z + num_samples_aa +
        num_levels_2d + num_levels_2d_array + num_levels_3d + num_levels_cube + num_levels_cube_array;
    return vec4<f32>(f32(sum));
}

[[group(1), binding(0)]]
var sampler_reg: sampler;

[[stage(fragment)]]
fn sample() -> [[location(0)]] vec4<f32> {
    let tc = vec2<f32>(0.5);
    let level = 2.3;
    let s2d = textureSample(image_2d, sampler_reg, tc);
    let s2d_offset = textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let s2d_level = textureSampleLevel(image_2d, sampler_reg, tc, level);
    let s2d_level_offset = textureSampleLevel(image_2d, sampler_reg, tc, level, vec2<i32>(3, 1));
    return s2d + s2d_offset + s2d_level + s2d_level_offset;
}

[[group(1), binding(1)]]
var sampler_cmp: sampler_comparison;
[[group(1), binding(2)]]
var image_2d_depth: texture_depth_2d;

[[stage(fragment)]]
fn sample_comparison() -> [[location(0)]] f32 {
    let tc = vec2<f32>(0.5);
    let dref = 0.5;
    let s2d_depth = textureSampleCompare(image_2d_depth, sampler_cmp, tc, dref);
    let s2d_depth_level = textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc, dref);
    return s2d_depth + s2d_depth_level;
}
