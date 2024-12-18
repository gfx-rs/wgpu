@group(0) @binding(0)
var image_mipmapped_src: texture_2d<u32>;
@group(0) @binding(3)
var image_multisampled_src: texture_multisampled_2d<u32>;
@group(0) @binding(4)
var image_depth_multisampled_src: texture_depth_multisampled_2d;
@group(0) @binding(1)
var image_storage_src: texture_storage_2d<rgba8uint, read>;
@group(0) @binding(5)
var image_array_src: texture_2d_array<u32>;
@group(0) @binding(6)
var image_dup_src: texture_storage_1d<r32uint,read>; // for #1307
@group(0) @binding(7)
var image_1d_src: texture_1d<u32>;
@group(0) @binding(2)
var image_dst: texture_storage_1d<r32uint,write>;

@compute @workgroup_size(16)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let dim = textureDimensions(image_storage_src);
    let itc = vec2<i32>(dim * local_id.xy) % vec2<i32>(10, 20);
    // loads with ivec2 coords.
    let value1 = textureLoad(image_mipmapped_src, itc, i32(local_id.z));
    let value2 = textureLoad(image_multisampled_src, itc, i32(local_id.z));
    let value4 = textureLoad(image_storage_src, itc);
    let value5 = textureLoad(image_array_src, itc, local_id.z, i32(local_id.z) + 1);
    let value6 = textureLoad(image_array_src, itc, i32(local_id.z), i32(local_id.z) + 1);
    let value7 = textureLoad(image_1d_src, i32(local_id.x), i32(local_id.z));
    // loads with uvec2 coords.
    let value1u = textureLoad(image_mipmapped_src, vec2<u32>(itc), i32(local_id.z));
    let value2u = textureLoad(image_multisampled_src, vec2<u32>(itc), i32(local_id.z));
    let value4u = textureLoad(image_storage_src, vec2<u32>(itc));
    let value5u = textureLoad(image_array_src, vec2<u32>(itc), local_id.z, i32(local_id.z) + 1);
    let value6u = textureLoad(image_array_src, vec2<u32>(itc), i32(local_id.z), i32(local_id.z) + 1);
    let value7u = textureLoad(image_1d_src, u32(local_id.x), i32(local_id.z));
    // store with ivec2 coords.
    textureStore(image_dst, itc.x, value1 + value2 + value4 + value5 + value6);
    // store with uvec2 coords.
    textureStore(image_dst, u32(itc.x), value1u + value2u + value4u + value5u + value6u);
}

@compute @workgroup_size(16, 1, 1)
fn depth_load(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let dim: vec2<u32> = textureDimensions(image_storage_src);
    let itc: vec2<i32> = (vec2<i32>(dim * local_id.xy) % vec2<i32>(10, 20));
    let val: f32 = textureLoad(image_depth_multisampled_src, itc, i32(local_id.z));
    textureStore(image_dst, itc.x, vec4<u32>(u32(val)));
    return;
}

@group(0) @binding(0)
var image_1d: texture_1d<f32>;
@group(0) @binding(1)
var image_2d: texture_2d<f32>;
@group(0) @binding(2)
var image_2d_u32: texture_2d<u32>;
@group(0) @binding(3)
var image_2d_i32: texture_2d<i32>;
@group(0) @binding(4)
var image_2d_array: texture_2d_array<f32>;
@group(0) @binding(5)
var image_cube: texture_cube<f32>;
@group(0) @binding(6)
var image_cube_array: texture_cube_array<f32>;
@group(0) @binding(7)
var image_3d: texture_3d<f32>;
@group(0) @binding(8)
var image_aa: texture_multisampled_2d<f32>;

@vertex
fn queries() -> @builtin(position) vec4<f32> {
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
    let dim_2s_ms = textureDimensions(image_aa);

    let sum = dim_1d + dim_2d.y + dim_2d_lod.y + dim_2d_array.y + dim_2d_array_lod.y + 
        dim_cube.y + dim_cube_lod.y + dim_cube_array.y + dim_cube_array_lod.y +
        dim_3d.z + dim_3d_lod.z;
    return vec4<f32>(f32(sum));
}

@vertex
fn levels_queries() -> @builtin(position) vec4<f32> {
    let num_levels_2d = textureNumLevels(image_2d);
    let num_layers_2d = textureNumLayers(image_2d_array);
    let num_levels_2d_array = textureNumLevels(image_2d_array);
    let num_layers_2d_array = textureNumLayers(image_2d_array);
    let num_levels_cube = textureNumLevels(image_cube);
    let num_levels_cube_array = textureNumLevels(image_cube_array);
    let num_layers_cube = textureNumLayers(image_cube_array);
    let num_levels_3d = textureNumLevels(image_3d);
    let num_samples_aa = textureNumSamples(image_aa);

    let sum = num_layers_2d + num_layers_cube + num_samples_aa +
        num_levels_2d + num_levels_2d_array + num_levels_3d + num_levels_cube + num_levels_cube_array;
    return vec4<f32>(f32(sum));
}

@group(1) @binding(0)
var sampler_reg: sampler;

@fragment
fn texture_sample() -> @location(0) vec4<f32> {
    let tc = vec2<f32>(0.5);
    let tc3 = vec3<f32>(0.5);
    let level = 2.3;
    var a: vec4<f32>;
    a += textureSample(image_1d, sampler_reg, tc.x);
    a += textureSample(image_2d, sampler_reg, tc);
    a += textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    a += textureSampleLevel(image_2d, sampler_reg, tc, level);
    a += textureSampleLevel(image_2d, sampler_reg, tc, level, vec2<i32>(3, 1));
    a += textureSampleBias(image_2d, sampler_reg, tc, 2.0, vec2<i32>(3, 1));
    a += textureSample(image_2d_array, sampler_reg, tc, 0u);
    a += textureSample(image_2d_array, sampler_reg, tc, 0u, vec2<i32>(3, 1));
    a += textureSampleLevel(image_2d_array, sampler_reg, tc, 0u, level);
    a += textureSampleLevel(image_2d_array, sampler_reg, tc, 0u, level, vec2<i32>(3, 1));
    a += textureSampleBias(image_2d_array, sampler_reg, tc, 0u, 2.0, vec2<i32>(3, 1));
    a += textureSample(image_2d_array, sampler_reg, tc, 0);
    a += textureSample(image_2d_array, sampler_reg, tc, 0, vec2<i32>(3, 1));
    a += textureSampleLevel(image_2d_array, sampler_reg, tc, 0, level);
    a += textureSampleLevel(image_2d_array, sampler_reg, tc, 0, level, vec2<i32>(3, 1));
    a += textureSampleBias(image_2d_array, sampler_reg, tc, 0, 2.0, vec2<i32>(3, 1));
    a += textureSample(image_cube_array, sampler_reg, tc3, 0u);
    a += textureSampleLevel(image_cube_array, sampler_reg, tc3, 0u, level);
    a += textureSampleBias(image_cube_array, sampler_reg, tc3, 0u, 2.0);
    a += textureSample(image_cube_array, sampler_reg, tc3, 0);
    a += textureSampleLevel(image_cube_array, sampler_reg, tc3, 0, level);
    a += textureSampleBias(image_cube_array, sampler_reg, tc3, 0, 2.0);
    return a;
}

@group(1) @binding(1)
var sampler_cmp: sampler_comparison;
@group(1) @binding(2)
var image_2d_depth: texture_depth_2d;
@group(1) @binding(3)
var image_2d_array_depth: texture_depth_2d_array;
@group(1) @binding(4)
var image_cube_depth: texture_depth_cube;

@fragment
fn texture_sample_comparison() -> @location(0) f32 {
    let tc = vec2<f32>(0.5);
    let tc3 = vec3<f32>(0.5);
    let dref = 0.5;
    var a: f32;
    a += textureSampleCompare(image_2d_depth, sampler_cmp, tc, dref);
    a += textureSampleCompare(image_2d_array_depth, sampler_cmp, tc, 0u, dref);
    a += textureSampleCompare(image_2d_array_depth, sampler_cmp, tc, 0, dref);
    a += textureSampleCompare(image_cube_depth, sampler_cmp, tc3, dref);
    a += textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc, dref);
    a += textureSampleCompareLevel(image_2d_array_depth, sampler_cmp, tc, 0u, dref);
    a += textureSampleCompareLevel(image_2d_array_depth, sampler_cmp, tc, 0, dref);
    a += textureSampleCompareLevel(image_cube_depth, sampler_cmp, tc3, dref);
    return a;
}

@fragment
fn gather() -> @location(0) vec4<f32> {
    let tc = vec2<f32>(0.5);
    let dref = 0.5;
    let s2d = textureGather(1, image_2d, sampler_reg, tc);
    let s2d_offset = textureGather(3, image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let s2d_depth = textureGatherCompare(image_2d_depth, sampler_cmp, tc, dref);
    let s2d_depth_offset = textureGatherCompare(image_2d_depth, sampler_cmp, tc, dref, vec2<i32>(3, 1));

    let u = textureGather(0, image_2d_u32, sampler_reg, tc);
    let i = textureGather(0, image_2d_i32, sampler_reg, tc);
    let f = vec4<f32>(u) + vec4<f32>(i);

    return s2d + s2d_offset + s2d_depth + s2d_depth_offset + f;
}

@fragment
fn depth_no_comparison() -> @location(0) vec4<f32> {
    let tc = vec2<f32>(0.5);
    let level = 1;
    let s2d = textureSample(image_2d_depth, sampler_reg, tc);
    let s2d_gather = textureGather(image_2d_depth, sampler_reg, tc);
    let s2d_level = textureSampleLevel(image_2d_depth, sampler_reg, tc, level);
    return s2d + s2d_gather + s2d_level;
}
