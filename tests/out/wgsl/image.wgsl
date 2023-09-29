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
    let _e9 = textureSample(image_1d, sampler_reg, tc.x);
    let _e10 = a;
    a = (_e10 + _e9);
    let _e14 = textureSample(image_2d, sampler_reg, tc);
    let _e15 = a;
    a = (_e15 + _e14);
    let _e19 = textureSample(image_2d, sampler_reg, tc, vec2<i32>(3, 1));
    let _e20 = a;
    a = (_e20 + _e19);
    let _e24 = textureSampleLevel(image_2d, sampler_reg, tc, 2.3);
    let _e25 = a;
    a = (_e25 + _e24);
    let _e29 = textureSampleLevel(image_2d, sampler_reg, tc, 2.3, vec2<i32>(3, 1));
    let _e30 = a;
    a = (_e30 + _e29);
    let _e35 = textureSampleBias(image_2d, sampler_reg, tc, 2.0, vec2<i32>(3, 1));
    let _e36 = a;
    a = (_e36 + _e35);
    let _e41 = textureSample(image_2d_array, sampler_reg, tc, 0u);
    let _e42 = a;
    a = (_e42 + _e41);
    let _e47 = textureSample(image_2d_array, sampler_reg, tc, 0u, vec2<i32>(3, 1));
    let _e48 = a;
    a = (_e48 + _e47);
    let _e53 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0u, 2.3);
    let _e54 = a;
    a = (_e54 + _e53);
    let _e59 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0u, 2.3, vec2<i32>(3, 1));
    let _e60 = a;
    a = (_e60 + _e59);
    let _e66 = textureSampleBias(image_2d_array, sampler_reg, tc, 0u, 2.0, vec2<i32>(3, 1));
    let _e67 = a;
    a = (_e67 + _e66);
    let _e72 = textureSample(image_2d_array, sampler_reg, tc, 0);
    let _e73 = a;
    a = (_e73 + _e72);
    let _e78 = textureSample(image_2d_array, sampler_reg, tc, 0, vec2<i32>(3, 1));
    let _e79 = a;
    a = (_e79 + _e78);
    let _e84 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0, 2.3);
    let _e85 = a;
    a = (_e85 + _e84);
    let _e90 = textureSampleLevel(image_2d_array, sampler_reg, tc, 0, 2.3, vec2<i32>(3, 1));
    let _e91 = a;
    a = (_e91 + _e90);
    let _e97 = textureSampleBias(image_2d_array, sampler_reg, tc, 0, 2.0, vec2<i32>(3, 1));
    let _e98 = a;
    a = (_e98 + _e97);
    let _e103 = textureSample(image_cube_array, sampler_reg, tc3_, 0u);
    let _e104 = a;
    a = (_e104 + _e103);
    let _e109 = textureSampleLevel(image_cube_array, sampler_reg, tc3_, 0u, 2.3);
    let _e110 = a;
    a = (_e110 + _e109);
    let _e116 = textureSampleBias(image_cube_array, sampler_reg, tc3_, 0u, 2.0);
    let _e117 = a;
    a = (_e117 + _e116);
    let _e122 = textureSample(image_cube_array, sampler_reg, tc3_, 0);
    let _e123 = a;
    a = (_e123 + _e122);
    let _e128 = textureSampleLevel(image_cube_array, sampler_reg, tc3_, 0, 2.3);
    let _e129 = a;
    a = (_e129 + _e128);
    let _e135 = textureSampleBias(image_cube_array, sampler_reg, tc3_, 0, 2.0);
    let _e136 = a;
    a = (_e136 + _e135);
    let _e138 = a;
    return _e138;
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
