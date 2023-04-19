@group(0) @binding(0) 
var image_mipmapped_src: texture_2d<u32>;
@group(0) @binding(3) 
var image_multisampled_src: texture_multisampled_2d<u32>;
@group(0) @binding(4) 
var image_depth_multisampled_src: texture_depth_multisampled_2d;
@group(0) @binding(1) 
var image_storage_src: texture_storage_2d<rgba8uint,read>;
@group(0) @binding(5) 
var image_array_src: texture_2d_array<u32>;
@group(0) @binding(6) 
var image_dup_src: texture_storage_1d<r32uint,read>;
@group(0) @binding(7) 
var image_1d_src: texture_1d<u32>;
@group(0) @binding(2) 
var image_dst: texture_storage_1d<r32uint,write>;
@group(0) @binding(0) 
var image_1d: texture_1d<f32>;
@group(0) @binding(1) 
var image_2d: texture_2d<f32>;
@group(0) @binding(2) 
var image_2d_u32_: texture_2d<u32>;
@group(0) @binding(3) 
var image_2d_i32_: texture_2d<i32>;
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
@group(1) @binding(0) 
var sampler_reg: sampler;
@group(1) @binding(1) 
var sampler_cmp: sampler_comparison;
@group(1) @binding(2) 
var image_2d_depth: texture_depth_2d;
@group(1) @binding(3) 
var image_2d_array_depth: texture_depth_2d_array;
@group(1) @binding(4) 
var image_cube_depth: texture_depth_cube;

@compute @workgroup_size(16, 1, 1) 
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let dim = textureDimensions(image_storage_src);
    let itc = (vec2<i32>((dim * local_id.xy)) % vec2<i32>(10, 20));
    let value1_ = textureLoad(image_mipmapped_src, itc, i32(local_id.z));
    let value2_ = textureLoad(image_multisampled_src, itc, i32(local_id.z));
    let value4_ = textureLoad(image_storage_src, itc);
    let value5_ = textureLoad(image_array_src, itc, local_id.z, (i32(local_id.z) + 1));
    let value6_ = textureLoad(image_array_src, itc, i32(local_id.z), (i32(local_id.z) + 1));
    let value7_ = textureLoad(image_1d_src, i32(local_id.x), i32(local_id.z));
    let value1u = textureLoad(image_mipmapped_src, vec2<u32>(itc), i32(local_id.z));
    let value2u = textureLoad(image_multisampled_src, vec2<u32>(itc), i32(local_id.z));
    let value4u = textureLoad(image_storage_src, vec2<u32>(itc));
    let value5u = textureLoad(image_array_src, vec2<u32>(itc), local_id.z, (i32(local_id.z) + 1));
    let value6u = textureLoad(image_array_src, vec2<u32>(itc), i32(local_id.z), (i32(local_id.z) + 1));
    let value7u = textureLoad(image_1d_src, u32(local_id.x), i32(local_id.z));
    textureStore(image_dst, itc.x, ((((value1_ + value2_) + value4_) + value5_) + value6_));
    textureStore(image_dst, u32(itc.x), ((((value1u + value2u) + value4u) + value5u) + value6u));
    return;
}

@compute @workgroup_size(16, 1, 1) 
fn depth_load(@builtin(local_invocation_id) local_id_1: vec3<u32>) {
    let dim_1 = textureDimensions(image_storage_src);
    let itc_1 = (vec2<i32>((dim_1 * local_id_1.xy)) % vec2<i32>(10, 20));
    let val = textureLoad(image_depth_multisampled_src, itc_1, i32(local_id_1.z));
    textureStore(image_dst, itc_1.x, vec4(u32(val)));
    return;
}

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
    let sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    return vec4(f32(sum));
}

@vertex 
fn levels_queries() -> @builtin(position) vec4<f32> {
    let num_levels_2d = textureNumLevels(image_2d);
    let num_levels_2d_array = textureNumLevels(image_2d_array);
    let num_layers_2d = textureNumLayers(image_2d_array);
    let num_levels_cube = textureNumLevels(image_cube);
    let num_levels_cube_array = textureNumLevels(image_cube_array);
    let num_layers_cube = textureNumLayers(image_cube_array);
    let num_levels_3d = textureNumLevels(image_3d);
    let num_samples_aa = textureNumSamples(image_aa);
    let sum_1 = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return vec4(f32(sum_1));
}

@fragment 
fn texture_sample() -> @location(0) vec4<f32> {
    var a: vec4<f32>;

    let tc = vec2(0.5);
    let tc3_ = vec3(0.5);
    let _e8 = textureSample(image_1d, sampler_reg, 0.5);
    let _e9 = a;
    a = (_e9 + _e8);
    let _e13 = textureSample(image_2d, sampler_reg, tc);
    let _e14 = a;
    a = (_e14 + _e13);
    let _e18 = textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let _e19 = a;
    a = (_e19 + _e18);
    let _e23 = textureSampleLevel(image_2d, sampler_reg, tc, 2.3);
    let _e24 = a;
    a = (_e24 + _e23);
    let _e28 = textureSampleLevel(image_2d, sampler_reg, tc, 2.3, vec2<i32>(3, 1));
    let _e29 = a;
    a = (_e29 + _e28);
    let _e34 = textureSampleBias(image_2d, sampler_reg, tc, 2.0, vec2<i32>(3, 1));
    let _e35 = a;
    a = (_e35 + _e34);
    let _e40 = textureSample(image_2d_array, sampler_reg, tc, 0u);
    let _e41 = a;
    a = (_e41 + _e40);
    let _e46 = textureSample(image_2d_array, sampler_reg, tc, 0u, vec2<i32>(3, 1));
    let _e47 = a;
    a = (_e47 + _e46);
    let _e52 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0u, 2.3);
    let _e53 = a;
    a = (_e53 + _e52);
    let _e58 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0u, 2.3, vec2<i32>(3, 1));
    let _e59 = a;
    a = (_e59 + _e58);
    let _e65 = textureSampleBias(image_2d_array, sampler_reg, tc, 0u, 2.0, vec2<i32>(3, 1));
    let _e66 = a;
    a = (_e66 + _e65);
    let _e71 = textureSample(image_2d_array, sampler_reg, tc, 0);
    let _e72 = a;
    a = (_e72 + _e71);
    let _e77 = textureSample(image_2d_array, sampler_reg, tc, 0, vec2<i32>(3, 1));
    let _e78 = a;
    a = (_e78 + _e77);
    let _e83 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0, 2.3);
    let _e84 = a;
    a = (_e84 + _e83);
    let _e89 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0, 2.3, vec2<i32>(3, 1));
    let _e90 = a;
    a = (_e90 + _e89);
    let _e96 = textureSampleBias(image_2d_array, sampler_reg, tc, 0, 2.0, vec2<i32>(3, 1));
    let _e97 = a;
    a = (_e97 + _e96);
    let _e102 = textureSample(image_cube_array, sampler_reg, tc3_, 0u);
    let _e103 = a;
    a = (_e103 + _e102);
    let _e108 = textureSampleLevel(image_cube_array, sampler_reg, tc3_, 0u, 2.3);
    let _e109 = a;
    a = (_e109 + _e108);
    let _e115 = textureSampleBias(image_cube_array, sampler_reg, tc3_, 0u, 2.0);
    let _e116 = a;
    a = (_e116 + _e115);
    let _e121 = textureSample(image_cube_array, sampler_reg, tc3_, 0);
    let _e122 = a;
    a = (_e122 + _e121);
    let _e127 = textureSampleLevel(image_cube_array, sampler_reg, tc3_, 0, 2.3);
    let _e128 = a;
    a = (_e128 + _e127);
    let _e134 = textureSampleBias(image_cube_array, sampler_reg, tc3_, 0, 2.0);
    let _e135 = a;
    a = (_e135 + _e134);
    let _e137 = a;
    return _e137;
}

@fragment 
fn texture_sample_comparison() -> @location(0) f32 {
    var a_1: f32;

    let tc_1 = vec2(0.5);
    let tc3_1 = vec3(0.5);
    let _e8 = textureSampleCompare(image_2d_depth, sampler_cmp, tc_1, 0.5);
    let _e9 = a_1;
    a_1 = (_e9 + _e8);
    let _e14 = textureSampleCompare(image_2d_array_depth, sampler_cmp, tc_1, 0u, 0.5);
    let _e15 = a_1;
    a_1 = (_e15 + _e14);
    let _e20 = textureSampleCompare(image_2d_array_depth, sampler_cmp, tc_1, 0, 0.5);
    let _e21 = a_1;
    a_1 = (_e21 + _e20);
    let _e25 = textureSampleCompare(image_cube_depth, sampler_cmp, tc3_1, 0.5);
    let _e26 = a_1;
    a_1 = (_e26 + _e25);
    let _e30 = textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc_1, 0.5);
    let _e31 = a_1;
    a_1 = (_e31 + _e30);
    let _e36 = textureSampleCompareLevel(image_2d_array_depth, sampler_cmp, tc_1, 0u, 0.5);
    let _e37 = a_1;
    a_1 = (_e37 + _e36);
    let _e42 = textureSampleCompareLevel(image_2d_array_depth, sampler_cmp, tc_1, 0, 0.5);
    let _e43 = a_1;
    a_1 = (_e43 + _e42);
    let _e47 = textureSampleCompareLevel(image_cube_depth, sampler_cmp, tc3_1, 0.5);
    let _e48 = a_1;
    a_1 = (_e48 + _e47);
    let _e50 = a_1;
    return _e50;
}

@fragment 
fn gather() -> @location(0) vec4<f32> {
    let tc_2 = vec2(0.5);
    let s2d = textureGather(1, image_2d, sampler_reg, tc_2);
    let s2d_offset = textureGather(3, image_2d, sampler_reg, tc_2, vec2<i32>(3, 1));
    let s2d_depth = textureGatherCompare(image_2d_depth, sampler_cmp, tc_2, 0.5);
    let s2d_depth_offset = textureGatherCompare(image_2d_depth, sampler_cmp, tc_2, 0.5, vec2<i32>(3, 1));
    let u = textureGather(0, image_2d_u32_, sampler_reg, tc_2);
    let i = textureGather(0, image_2d_i32_, sampler_reg, tc_2);
    let f = (vec4<f32>(u) + vec4<f32>(i));
    return ((((s2d + s2d_offset) + s2d_depth) + s2d_depth_offset) + f);
}

@fragment 
fn depth_no_comparison() -> @location(0) vec4<f32> {
    let tc_3 = vec2(0.5);
    let s2d_1 = textureSample(image_2d_depth, sampler_reg, tc_3);
    let s2d_gather = textureGather(image_2d_depth, sampler_reg, tc_3);
    return (vec4(s2d_1) + s2d_gather);
}
