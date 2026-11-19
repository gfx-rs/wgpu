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
    let itc = (vec2<i32>((dim * local_id.xy)) % vec2<i32>(10i, 20i));
    let value1_ = textureLoad(image_mipmapped_src, itc, i32(local_id.z));
    let value1_2_ = textureLoad(image_mipmapped_src, itc, u32(local_id.z));
    let value2_ = textureLoad(image_multisampled_src, itc, i32(local_id.z));
    let value3_ = textureLoad(image_multisampled_src, itc, u32(local_id.z));
    let value4_ = textureLoad(image_storage_src, itc);
    let value5_ = textureLoad(image_array_src, itc, local_id.z, (i32(local_id.z) + 1i));
    let value6_ = textureLoad(image_array_src, itc, i32(local_id.z), (i32(local_id.z) + 1i));
    let value7_ = textureLoad(image_1d_src, i32(local_id.x), i32(local_id.z));
    let value8_ = textureLoad(image_dup_src, i32(local_id.x));
    let value1u = textureLoad(image_mipmapped_src, vec2<u32>(itc), i32(local_id.z));
    let value2u = textureLoad(image_multisampled_src, vec2<u32>(itc), i32(local_id.z));
    let value3u = textureLoad(image_multisampled_src, vec2<u32>(itc), u32(local_id.z));
    let value4u = textureLoad(image_storage_src, vec2<u32>(itc));
    let value5u = textureLoad(image_array_src, vec2<u32>(itc), local_id.z, (i32(local_id.z) + 1i));
    let value6u = textureLoad(image_array_src, vec2<u32>(itc), i32(local_id.z), (i32(local_id.z) + 1i));
    let value7u = textureLoad(image_1d_src, u32(local_id.x), i32(local_id.z));
    textureStore(image_dst, itc.x, ((((value1_ + value2_) + value4_) + value5_) + value6_));
    textureStore(image_dst, u32(itc.x), ((((value1u + value2u) + value4u) + value5u) + value6u));
    return;
}

@compute @workgroup_size(16, 1, 1)
fn depth_load(@builtin(local_invocation_id) local_id_1: vec3<u32>) {
    let dim_1 = textureDimensions(image_storage_src);
    let itc_1 = (vec2<i32>((dim_1 * local_id_1.xy)) % vec2<i32>(10i, 20i));
    let val = textureLoad(image_depth_multisampled_src, itc_1, i32(local_id_1.z));
    textureStore(image_dst, itc_1.x, vec4(u32(val)));
    return;
}

@vertex
fn queries() -> @builtin(position) vec4<f32> {
    let dim_1d = textureDimensions(image_1d);
    let dim_1d_lod = textureDimensions(image_1d, i32(dim_1d));
    let dim_2d = textureDimensions(image_2d);
    let dim_2d_lod = textureDimensions(image_2d, 1i);
    let dim_2d_array = textureDimensions(image_2d_array);
    let dim_2d_array_lod = textureDimensions(image_2d_array, 1i);
    let dim_cube = textureDimensions(image_cube);
    let dim_cube_lod = textureDimensions(image_cube, 1i);
    let dim_cube_array = textureDimensions(image_cube_array);
    let dim_cube_array_lod = textureDimensions(image_cube_array, 1i);
    let dim_3d = textureDimensions(image_3d);
    let dim_3d_lod = textureDimensions(image_3d, 1i);
    let dim_2s_ms = textureDimensions(image_aa);
    let sum = ((((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z);
    return vec4(f32(sum));
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
    let sum_1 = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    return vec4(f32(sum_1));
}

@fragment
fn texture_sample() -> @location(0) vec4<f32> {
    var a: vec4<f32>;

    let _e1 = vec2(0.5f);
    let _e3 = vec3(0.5f);
    let _e6 = vec2<i32>(3i, 1i);
    let _e9 = a;
    let _e12 = textureSample(image_1d, sampler_reg, 0.5f);
    a = (_e9 + _e12);
    let _e14 = a;
    let _e17 = textureSample(image_2d, sampler_reg, _e1);
    a = (_e14 + _e17);
    let _e19 = a;
    let _e25 = textureSample(image_2d, sampler_reg, _e1, vec2<i32>(3i, 1i));
    a = (_e19 + _e25);
    let _e27 = a;
    let _e30 = textureSampleLevel(image_2d, sampler_reg, _e1, 2.3f);
    a = (_e27 + _e30);
    let _e32 = a;
    let _e35 = textureSampleLevel(image_2d, sampler_reg, _e1, 2.3f, vec2<i32>(3i, 1i));
    a = (_e32 + _e35);
    let _e37 = a;
    let _e41 = textureSampleBias(image_2d, sampler_reg, _e1, 2f, vec2<i32>(3i, 1i));
    a = (_e37 + _e41);
    let _e43 = a;
    let _e46 = textureSampleGrad(image_2d, sampler_reg, _e1, _e1, _e1);
    a = (_e43 + _e46);
    let _e48 = a;
    let _e51 = textureSampleBaseClampToEdge(image_2d, sampler_reg, _e1);
    a = (_e48 + _e51);
    let _e53 = a;
    let _e57 = textureSample(image_2d_array, sampler_reg, _e1, 0u);
    a = (_e53 + _e57);
    let _e59 = a;
    let _e63 = textureSample(image_2d_array, sampler_reg, _e1, 0u, vec2<i32>(3i, 1i));
    a = (_e59 + _e63);
    let _e65 = a;
    let _e69 = textureSampleLevel(image_2d_array, sampler_reg, _e1, 0u, 2.3f);
    a = (_e65 + _e69);
    let _e71 = a;
    let _e75 = textureSampleLevel(image_2d_array, sampler_reg, _e1, 0u, 2.3f, vec2<i32>(3i, 1i));
    a = (_e71 + _e75);
    let _e77 = a;
    let _e82 = textureSampleBias(image_2d_array, sampler_reg, _e1, 0u, 2f, vec2<i32>(3i, 1i));
    a = (_e77 + _e82);
    let _e84 = a;
    let _e88 = textureSample(image_2d_array, sampler_reg, _e1, 0i);
    a = (_e84 + _e88);
    let _e90 = a;
    let _e94 = textureSample(image_2d_array, sampler_reg, _e1, 0i, vec2<i32>(3i, 1i));
    a = (_e90 + _e94);
    let _e96 = a;
    let _e100 = textureSampleLevel(image_2d_array, sampler_reg, _e1, 0i, 2.3f);
    a = (_e96 + _e100);
    let _e102 = a;
    let _e106 = textureSampleLevel(image_2d_array, sampler_reg, _e1, 0i, 2.3f, vec2<i32>(3i, 1i));
    a = (_e102 + _e106);
    let _e108 = a;
    let _e113 = textureSampleBias(image_2d_array, sampler_reg, _e1, 0i, 2f, vec2<i32>(3i, 1i));
    a = (_e108 + _e113);
    let _e115 = a;
    let _e118 = textureSampleGrad(image_cube, sampler_reg, _e3, _e3, _e3);
    a = (_e115 + _e118);
    let _e120 = a;
    let _e124 = textureSample(image_cube_array, sampler_reg, _e3, 0u);
    a = (_e120 + _e124);
    let _e126 = a;
    let _e130 = textureSampleLevel(image_cube_array, sampler_reg, _e3, 0u, 2.3f);
    a = (_e126 + _e130);
    let _e132 = a;
    let _e137 = textureSampleBias(image_cube_array, sampler_reg, _e3, 0u, 2f);
    a = (_e132 + _e137);
    let _e139 = a;
    let _e143 = textureSampleGrad(image_cube_array, sampler_reg, _e3, 0u, _e3, _e3);
    a = (_e139 + _e143);
    let _e145 = a;
    let _e149 = textureSample(image_cube_array, sampler_reg, _e3, 0i);
    a = (_e145 + _e149);
    let _e151 = a;
    let _e155 = textureSampleLevel(image_cube_array, sampler_reg, _e3, 0i, 2.3f);
    a = (_e151 + _e155);
    let _e157 = a;
    let _e162 = textureSampleBias(image_cube_array, sampler_reg, _e3, 0i, 2f);
    a = (_e157 + _e162);
    let _e164 = a;
    let _e168 = textureSampleGrad(image_cube_array, sampler_reg, _e3, 0i, _e3, _e3);
    a = (_e164 + _e168);
    let _e170 = a;
    let _e173 = textureSampleGrad(image_3d, sampler_reg, _e3, _e3, _e3);
    a = (_e170 + _e173);
    let _e175 = a;
    return _e175;
}

@fragment
fn texture_sample_comparison() -> @location(0) f32 {
    var a_1: f32;

    let tc = vec2(0.5f);
    let tc3_ = vec3(0.5f);
    let _e6 = a_1;
    let _e9 = textureSampleCompare(image_2d_depth, sampler_cmp, tc, 0.5f);
    a_1 = (_e6 + _e9);
    let _e11 = a_1;
    let _e15 = textureSampleCompare(image_2d_array_depth, sampler_cmp, tc, 0u, 0.5f);
    a_1 = (_e11 + _e15);
    let _e17 = a_1;
    let _e21 = textureSampleCompare(image_2d_array_depth, sampler_cmp, tc, 0i, 0.5f);
    a_1 = (_e17 + _e21);
    let _e23 = a_1;
    let _e26 = textureSampleCompare(image_cube_depth, sampler_cmp, tc3_, 0.5f);
    a_1 = (_e23 + _e26);
    let _e28 = a_1;
    let _e31 = textureSampleCompareLevel(image_2d_depth, sampler_cmp, tc, 0.5f);
    a_1 = (_e28 + _e31);
    let _e33 = a_1;
    let _e37 = textureSampleCompareLevel(image_2d_array_depth, sampler_cmp, tc, 0u, 0.5f);
    a_1 = (_e33 + _e37);
    let _e39 = a_1;
    let _e43 = textureSampleCompareLevel(image_2d_array_depth, sampler_cmp, tc, 0i, 0.5f);
    a_1 = (_e39 + _e43);
    let _e45 = a_1;
    let _e48 = textureSampleCompareLevel(image_cube_depth, sampler_cmp, tc3_, 0.5f);
    a_1 = (_e45 + _e48);
    let _e50 = a_1;
    return _e50;
}

@fragment
fn gather() -> @location(0) vec4<f32> {
    let tc_1 = vec2(0.5f);
    let s2d = textureGather(1, image_2d, sampler_reg, tc_1);
    let s2d_offset = textureGather(3, image_2d, sampler_reg, tc_1, vec2<i32>(3i, 1i));
    let s2d_depth = textureGatherCompare(image_2d_depth, sampler_cmp, tc_1, 0.5f);
    let s2d_depth_offset = textureGatherCompare(image_2d_depth, sampler_cmp, tc_1, 0.5f, vec2<i32>(3i, 1i));
    let u = textureGather(0, image_2d_u32_, sampler_reg, tc_1);
    let i = textureGather(0, image_2d_i32_, sampler_reg, tc_1);
    let f = (vec4<f32>(u) + vec4<f32>(i));
    return ((((s2d + s2d_offset) + s2d_depth) + s2d_depth_offset) + f);
}

@fragment
fn depth_no_comparison() -> @location(0) vec4<f32> {
    let tc_2 = vec2(0.5f);
    let s2d_1 = textureSample(image_2d_depth, sampler_reg, tc_2);
    let s2d_gather = textureGather(image_2d_depth, sampler_reg, tc_2);
    let s2d_level = textureSampleLevel(image_2d_depth, sampler_reg, tc_2, 1i);
    return ((vec4(s2d_1) + s2d_gather) + vec4(s2d_level));
}
