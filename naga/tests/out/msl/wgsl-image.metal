// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

metal::int2 naga_mod(metal::int2 lhs, metal::int2 rhs) {
    metal::int2 divisor = metal::select(rhs, 1, (lhs == (-2147483647 - 1) & rhs == -1) | (rhs == 0));
    return lhs - (lhs / divisor) * divisor;
}


struct main_Input {
};
kernel void main_(
  metal::uint3 local_id [[thread_position_in_threadgroup]]
, metal::texture2d<uint, metal::access::sample> image_mipmapped_src [[user(fake0)]]
, metal::texture2d_ms<uint, metal::access::read> image_multisampled_src [[user(fake0)]]
, metal::texture2d<uint, metal::access::read> image_storage_src [[user(fake0)]]
, metal::texture2d_array<uint, metal::access::sample> image_array_src [[user(fake0)]]
, metal::texture1d<uint, metal::access::read> image_dup_src [[user(fake0)]]
, metal::texture1d<uint, metal::access::sample> image_1d_src [[user(fake0)]]
, metal::texture1d<uint, metal::access::write> image_dst [[user(fake0)]]
) {
    metal::uint2 dim = metal::uint2(image_storage_src.get_width(), image_storage_src.get_height());
    metal::int2 itc = naga_mod(static_cast<metal::int2>(dim * local_id.xy), metal::int2(10, 20));
    metal::uint4 value1_ = image_mipmapped_src.read(metal::uint2(itc), static_cast<int>(local_id.z));
    metal::uint4 value1_2_ = image_mipmapped_src.read(metal::uint2(itc), static_cast<uint>(local_id.z));
    metal::uint4 value2_ = image_multisampled_src.read(metal::uint2(itc), static_cast<int>(local_id.z));
    metal::uint4 value3_ = image_multisampled_src.read(metal::uint2(itc), static_cast<uint>(local_id.z));
    metal::uint4 value4_ = image_storage_src.read(metal::uint2(itc));
    metal::uint4 value5_ = image_array_src.read(metal::uint2(itc), local_id.z, as_type<int>(as_type<uint>(static_cast<int>(local_id.z)) + as_type<uint>(1)));
    metal::uint4 value6_ = image_array_src.read(metal::uint2(itc), static_cast<int>(local_id.z), as_type<int>(as_type<uint>(static_cast<int>(local_id.z)) + as_type<uint>(1)));
    metal::uint4 value7_ = image_1d_src.read(uint(static_cast<int>(local_id.x)));
    metal::uint4 value8_ = image_dup_src.read(uint(static_cast<int>(local_id.x)));
    metal::uint4 value1u = image_mipmapped_src.read(metal::uint2(static_cast<metal::uint2>(itc)), static_cast<int>(local_id.z));
    metal::uint4 value2u = image_multisampled_src.read(metal::uint2(static_cast<metal::uint2>(itc)), static_cast<int>(local_id.z));
    metal::uint4 value3u = image_multisampled_src.read(metal::uint2(static_cast<metal::uint2>(itc)), static_cast<uint>(local_id.z));
    metal::uint4 value4u = image_storage_src.read(metal::uint2(static_cast<metal::uint2>(itc)));
    metal::uint4 value5u = image_array_src.read(metal::uint2(static_cast<metal::uint2>(itc)), local_id.z, as_type<int>(as_type<uint>(static_cast<int>(local_id.z)) + as_type<uint>(1)));
    metal::uint4 value6u = image_array_src.read(metal::uint2(static_cast<metal::uint2>(itc)), static_cast<int>(local_id.z), as_type<int>(as_type<uint>(static_cast<int>(local_id.z)) + as_type<uint>(1)));
    metal::uint4 value7u = image_1d_src.read(uint(static_cast<uint>(local_id.x)));
    image_dst.write((((value1_ + value2_) + value4_) + value5_) + value6_, uint(itc.x));
    image_dst.write((((value1u + value2u) + value4u) + value5u) + value6u, uint(static_cast<uint>(itc.x)));
    return;
}

uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}


struct depth_loadInput {
};
kernel void depth_load(
  metal::uint3 local_id_1 [[thread_position_in_threadgroup]]
, metal::depth2d_ms<float, metal::access::read> image_depth_multisampled_src [[user(fake0)]]
, metal::texture2d<uint, metal::access::read> image_storage_src [[user(fake0)]]
, metal::texture1d<uint, metal::access::write> image_dst [[user(fake0)]]
) {
    metal::uint2 dim_1 = metal::uint2(image_storage_src.get_width(), image_storage_src.get_height());
    metal::int2 itc_1 = naga_mod(static_cast<metal::int2>(dim_1 * local_id_1.xy), metal::int2(10, 20));
    float val = image_depth_multisampled_src.read(metal::uint2(itc_1), static_cast<int>(local_id_1.z));
    image_dst.write(metal::uint4(naga_f2u32(val)), uint(itc_1.x));
    return;
}


struct queriesOutput {
    metal::float4 member_2 [[position]];
};
vertex queriesOutput queries(
  metal::texture1d<float, metal::access::sample> image_1d [[user(fake0)]]
, metal::texture2d<float, metal::access::sample> image_2d [[user(fake0)]]
, metal::texture2d_array<float, metal::access::sample> image_2d_array [[user(fake0)]]
, metal::texturecube<float, metal::access::sample> image_cube [[user(fake0)]]
, metal::texturecube_array<float, metal::access::sample> image_cube_array [[user(fake0)]]
, metal::texture3d<float, metal::access::sample> image_3d [[user(fake0)]]
, metal::texture2d_ms<float, metal::access::read> image_aa [[user(fake0)]]
) {
    uint dim_1d = image_1d.get_width();
    uint dim_1d_lod = image_1d.get_width();
    metal::uint2 dim_2d = metal::uint2(image_2d.get_width(), image_2d.get_height());
    metal::uint2 dim_2d_lod = metal::uint2(image_2d.get_width(1), image_2d.get_height(1));
    metal::uint2 dim_2d_array = metal::uint2(image_2d_array.get_width(), image_2d_array.get_height());
    metal::uint2 dim_2d_array_lod = metal::uint2(image_2d_array.get_width(1), image_2d_array.get_height(1));
    metal::uint2 dim_cube = metal::uint2(image_cube.get_width());
    metal::uint2 dim_cube_lod = metal::uint2(image_cube.get_width(1));
    metal::uint2 dim_cube_array = metal::uint2(image_cube_array.get_width());
    metal::uint2 dim_cube_array_lod = metal::uint2(image_cube_array.get_width(1));
    metal::uint3 dim_3d = metal::uint3(image_3d.get_width(), image_3d.get_height(), image_3d.get_depth());
    metal::uint3 dim_3d_lod = metal::uint3(image_3d.get_width(1), image_3d.get_height(1), image_3d.get_depth(1));
    metal::uint2 dim_2s_ms = metal::uint2(image_aa.get_width(), image_aa.get_height());
    uint sum = (((((((((dim_1d + dim_2d.y) + dim_2d_lod.y) + dim_2d_array.y) + dim_2d_array_lod.y) + dim_cube.y) + dim_cube_lod.y) + dim_cube_array.y) + dim_cube_array_lod.y) + dim_3d.z) + dim_3d_lod.z;
    return queriesOutput { metal::float4(static_cast<float>(sum)) };
}


struct levels_queriesOutput {
    metal::float4 member_3 [[position]];
};
vertex levels_queriesOutput levels_queries(
  metal::texture2d<float, metal::access::sample> image_2d [[user(fake0)]]
, metal::texture2d_array<float, metal::access::sample> image_2d_array [[user(fake0)]]
, metal::texturecube<float, metal::access::sample> image_cube [[user(fake0)]]
, metal::texturecube_array<float, metal::access::sample> image_cube_array [[user(fake0)]]
, metal::texture3d<float, metal::access::sample> image_3d [[user(fake0)]]
, metal::texture2d_ms<float, metal::access::read> image_aa [[user(fake0)]]
) {
    uint num_levels_2d = image_2d.get_num_mip_levels();
    uint num_layers_2d = image_2d_array.get_array_size();
    uint num_levels_2d_array = image_2d_array.get_num_mip_levels();
    uint num_layers_2d_array = image_2d_array.get_array_size();
    uint num_levels_cube = image_cube.get_num_mip_levels();
    uint num_levels_cube_array = image_cube_array.get_num_mip_levels();
    uint num_layers_cube = image_cube_array.get_array_size();
    uint num_levels_3d = image_3d.get_num_mip_levels();
    uint num_samples_aa = image_aa.get_num_samples();
    uint sum_1 = ((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array;
    return levels_queriesOutput { metal::float4(static_cast<float>(sum_1)) };
}

metal::float4 nagaTextureSampleBaseClampToEdge(metal::texture2d<float, metal::access::sample> tex, metal::sampler samp, metal::float2 coords) {
    metal::float2 half_texel = 0.5 / metal::float2(tex.get_width(0u), tex.get_height(0u));
    return tex.sample(samp, metal::clamp(coords, half_texel, 1.0 - half_texel), metal::level(0.0));
}


struct texture_sampleOutput {
    metal::float4 member_4 [[color(0)]];
};
fragment texture_sampleOutput texture_sample(
  metal::texture1d<float, metal::access::sample> image_1d [[user(fake0)]]
, metal::texture2d<float, metal::access::sample> image_2d [[user(fake0)]]
, metal::texture2d_array<float, metal::access::sample> image_2d_array [[user(fake0)]]
, metal::texturecube_array<float, metal::access::sample> image_cube_array [[user(fake0)]]
, metal::sampler sampler_reg [[user(fake0)]]
) {
    metal::float4 a = {};
    metal::float2 _e1 = metal::float2(0.5);
    metal::float3 _e3 = metal::float3(0.5);
    metal::int2 _e6 = metal::int2(3, 1);
    metal::float4 _e9 = a;
    metal::float4 _e12 = image_1d.sample(sampler_reg, 0.5);
    a = _e9 + _e12;
    metal::float4 _e14 = a;
    metal::float4 _e17 = image_2d.sample(sampler_reg, _e1);
    a = _e14 + _e17;
    metal::float4 _e19 = a;
    metal::float4 _e25 = image_2d.sample(sampler_reg, _e1, metal::int2(3, 1));
    a = _e19 + _e25;
    metal::float4 _e27 = a;
    metal::float4 _e30 = image_2d.sample(sampler_reg, _e1, metal::level(2.3));
    a = _e27 + _e30;
    metal::float4 _e32 = a;
    metal::float4 _e35 = image_2d.sample(sampler_reg, _e1, metal::level(2.3), _e6);
    a = _e32 + _e35;
    metal::float4 _e37 = a;
    metal::float4 _e41 = image_2d.sample(sampler_reg, _e1, metal::bias(2.0), _e6);
    a = _e37 + _e41;
    metal::float4 _e43 = a;
    metal::float4 _e46 = nagaTextureSampleBaseClampToEdge(image_2d, sampler_reg, _e1);
    a = _e43 + _e46;
    metal::float4 _e48 = a;
    metal::float4 _e52 = image_2d_array.sample(sampler_reg, _e1, 0u);
    a = _e48 + _e52;
    metal::float4 _e54 = a;
    metal::float4 _e58 = image_2d_array.sample(sampler_reg, _e1, 0u, _e6);
    a = _e54 + _e58;
    metal::float4 _e60 = a;
    metal::float4 _e64 = image_2d_array.sample(sampler_reg, _e1, 0u, metal::level(2.3));
    a = _e60 + _e64;
    metal::float4 _e66 = a;
    metal::float4 _e70 = image_2d_array.sample(sampler_reg, _e1, 0u, metal::level(2.3), _e6);
    a = _e66 + _e70;
    metal::float4 _e72 = a;
    metal::float4 _e77 = image_2d_array.sample(sampler_reg, _e1, 0u, metal::bias(2.0), _e6);
    a = _e72 + _e77;
    metal::float4 _e79 = a;
    metal::float4 _e83 = image_2d_array.sample(sampler_reg, _e1, 0);
    a = _e79 + _e83;
    metal::float4 _e85 = a;
    metal::float4 _e89 = image_2d_array.sample(sampler_reg, _e1, 0, _e6);
    a = _e85 + _e89;
    metal::float4 _e91 = a;
    metal::float4 _e95 = image_2d_array.sample(sampler_reg, _e1, 0, metal::level(2.3));
    a = _e91 + _e95;
    metal::float4 _e97 = a;
    metal::float4 _e101 = image_2d_array.sample(sampler_reg, _e1, 0, metal::level(2.3), _e6);
    a = _e97 + _e101;
    metal::float4 _e103 = a;
    metal::float4 _e108 = image_2d_array.sample(sampler_reg, _e1, 0, metal::bias(2.0), _e6);
    a = _e103 + _e108;
    metal::float4 _e110 = a;
    metal::float4 _e114 = image_cube_array.sample(sampler_reg, _e3, 0u);
    a = _e110 + _e114;
    metal::float4 _e116 = a;
    metal::float4 _e120 = image_cube_array.sample(sampler_reg, _e3, 0u, metal::level(2.3));
    a = _e116 + _e120;
    metal::float4 _e122 = a;
    metal::float4 _e127 = image_cube_array.sample(sampler_reg, _e3, 0u, metal::bias(2.0));
    a = _e122 + _e127;
    metal::float4 _e129 = a;
    metal::float4 _e133 = image_cube_array.sample(sampler_reg, _e3, 0);
    a = _e129 + _e133;
    metal::float4 _e135 = a;
    metal::float4 _e139 = image_cube_array.sample(sampler_reg, _e3, 0, metal::level(2.3));
    a = _e135 + _e139;
    metal::float4 _e141 = a;
    metal::float4 _e146 = image_cube_array.sample(sampler_reg, _e3, 0, metal::bias(2.0));
    a = _e141 + _e146;
    metal::float4 _e148 = a;
    return texture_sampleOutput { _e148 };
}


struct texture_sample_comparisonOutput {
    float member_5 [[color(0)]];
};
fragment texture_sample_comparisonOutput texture_sample_comparison(
  metal::sampler sampler_cmp [[user(fake0)]]
, metal::depth2d<float, metal::access::sample> image_2d_depth [[user(fake0)]]
, metal::depth2d_array<float, metal::access::sample> image_2d_array_depth [[user(fake0)]]
, metal::depthcube<float, metal::access::sample> image_cube_depth [[user(fake0)]]
) {
    float a_1 = {};
    metal::float2 tc = metal::float2(0.5);
    metal::float3 tc3_ = metal::float3(0.5);
    float _e6 = a_1;
    float _e9 = image_2d_depth.sample_compare(sampler_cmp, tc, 0.5);
    a_1 = _e6 + _e9;
    float _e11 = a_1;
    float _e15 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0u, 0.5);
    a_1 = _e11 + _e15;
    float _e17 = a_1;
    float _e21 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0, 0.5);
    a_1 = _e17 + _e21;
    float _e23 = a_1;
    float _e26 = image_cube_depth.sample_compare(sampler_cmp, tc3_, 0.5);
    a_1 = _e23 + _e26;
    float _e28 = a_1;
    float _e31 = image_2d_depth.sample_compare(sampler_cmp, tc, 0.5);
    a_1 = _e28 + _e31;
    float _e33 = a_1;
    float _e37 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0u, 0.5);
    a_1 = _e33 + _e37;
    float _e39 = a_1;
    float _e43 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0, 0.5);
    a_1 = _e39 + _e43;
    float _e45 = a_1;
    float _e48 = image_cube_depth.sample_compare(sampler_cmp, tc3_, 0.5);
    a_1 = _e45 + _e48;
    float _e50 = a_1;
    return texture_sample_comparisonOutput { _e50 };
}


struct gatherOutput {
    metal::float4 member_6 [[color(0)]];
};
fragment gatherOutput gather(
  metal::texture2d<float, metal::access::sample> image_2d [[user(fake0)]]
, metal::texture2d<uint, metal::access::sample> image_2d_u32_ [[user(fake0)]]
, metal::texture2d<int, metal::access::sample> image_2d_i32_ [[user(fake0)]]
, metal::sampler sampler_reg [[user(fake0)]]
, metal::sampler sampler_cmp [[user(fake0)]]
, metal::depth2d<float, metal::access::sample> image_2d_depth [[user(fake0)]]
) {
    metal::float2 tc_1 = metal::float2(0.5);
    metal::float4 s2d = image_2d.gather(sampler_reg, tc_1, metal::int2(0), metal::component::y);
    metal::float4 s2d_offset = image_2d.gather(sampler_reg, tc_1, metal::int2(3, 1), metal::component::w);
    metal::float4 s2d_depth = image_2d_depth.gather_compare(sampler_cmp, tc_1, 0.5);
    metal::float4 s2d_depth_offset = image_2d_depth.gather_compare(sampler_cmp, tc_1, 0.5, metal::int2(3, 1));
    metal::uint4 u = image_2d_u32_.gather(sampler_reg, tc_1);
    metal::int4 i = image_2d_i32_.gather(sampler_reg, tc_1);
    metal::float4 f = static_cast<metal::float4>(u) + static_cast<metal::float4>(i);
    return gatherOutput { (((s2d + s2d_offset) + s2d_depth) + s2d_depth_offset) + f };
}


struct depth_no_comparisonOutput {
    metal::float4 member_7 [[color(0)]];
};
fragment depth_no_comparisonOutput depth_no_comparison(
  metal::sampler sampler_reg [[user(fake0)]]
, metal::depth2d<float, metal::access::sample> image_2d_depth [[user(fake0)]]
) {
    metal::float2 tc_2 = metal::float2(0.5);
    float s2d_1 = image_2d_depth.sample(sampler_reg, tc_2);
    metal::float4 s2d_gather = image_2d_depth.gather(sampler_reg, tc_2);
    float s2d_level = image_2d_depth.sample(sampler_reg, tc_2, metal::level(1));
    return depth_no_comparisonOutput { (metal::float4(s2d_1) + s2d_gather) + metal::float4(s2d_level) };
}
