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
    metal::float4 _e11 = image_1d.sample(sampler_reg, 0.5);
    metal::float4 _e12 = a;
    a = _e12 + _e11;
    metal::float4 _e16 = image_2d.sample(sampler_reg, _e1);
    metal::float4 _e17 = a;
    a = _e17 + _e16;
    metal::float4 _e24 = image_2d.sample(sampler_reg, _e1, metal::int2(3, 1));
    metal::float4 _e25 = a;
    a = _e25 + _e24;
    metal::float4 _e29 = image_2d.sample(sampler_reg, _e1, metal::level(2.3));
    metal::float4 _e30 = a;
    a = _e30 + _e29;
    metal::float4 _e34 = image_2d.sample(sampler_reg, _e1, metal::level(2.3), _e6);
    metal::float4 _e35 = a;
    a = _e35 + _e34;
    metal::float4 _e40 = image_2d.sample(sampler_reg, _e1, metal::bias(2.0), _e6);
    metal::float4 _e41 = a;
    a = _e41 + _e40;
    metal::float4 _e45 = nagaTextureSampleBaseClampToEdge(image_2d, sampler_reg, _e1);
    metal::float4 _e46 = a;
    a = _e46 + _e45;
    metal::float4 _e51 = image_2d_array.sample(sampler_reg, _e1, 0u);
    metal::float4 _e52 = a;
    a = _e52 + _e51;
    metal::float4 _e57 = image_2d_array.sample(sampler_reg, _e1, 0u, _e6);
    metal::float4 _e58 = a;
    a = _e58 + _e57;
    metal::float4 _e63 = image_2d_array.sample(sampler_reg, _e1, 0u, metal::level(2.3));
    metal::float4 _e64 = a;
    a = _e64 + _e63;
    metal::float4 _e69 = image_2d_array.sample(sampler_reg, _e1, 0u, metal::level(2.3), _e6);
    metal::float4 _e70 = a;
    a = _e70 + _e69;
    metal::float4 _e76 = image_2d_array.sample(sampler_reg, _e1, 0u, metal::bias(2.0), _e6);
    metal::float4 _e77 = a;
    a = _e77 + _e76;
    metal::float4 _e82 = image_2d_array.sample(sampler_reg, _e1, 0);
    metal::float4 _e83 = a;
    a = _e83 + _e82;
    metal::float4 _e88 = image_2d_array.sample(sampler_reg, _e1, 0, _e6);
    metal::float4 _e89 = a;
    a = _e89 + _e88;
    metal::float4 _e94 = image_2d_array.sample(sampler_reg, _e1, 0, metal::level(2.3));
    metal::float4 _e95 = a;
    a = _e95 + _e94;
    metal::float4 _e100 = image_2d_array.sample(sampler_reg, _e1, 0, metal::level(2.3), _e6);
    metal::float4 _e101 = a;
    a = _e101 + _e100;
    metal::float4 _e107 = image_2d_array.sample(sampler_reg, _e1, 0, metal::bias(2.0), _e6);
    metal::float4 _e108 = a;
    a = _e108 + _e107;
    metal::float4 _e113 = image_cube_array.sample(sampler_reg, _e3, 0u);
    metal::float4 _e114 = a;
    a = _e114 + _e113;
    metal::float4 _e119 = image_cube_array.sample(sampler_reg, _e3, 0u, metal::level(2.3));
    metal::float4 _e120 = a;
    a = _e120 + _e119;
    metal::float4 _e126 = image_cube_array.sample(sampler_reg, _e3, 0u, metal::bias(2.0));
    metal::float4 _e127 = a;
    a = _e127 + _e126;
    metal::float4 _e132 = image_cube_array.sample(sampler_reg, _e3, 0);
    metal::float4 _e133 = a;
    a = _e133 + _e132;
    metal::float4 _e138 = image_cube_array.sample(sampler_reg, _e3, 0, metal::level(2.3));
    metal::float4 _e139 = a;
    a = _e139 + _e138;
    metal::float4 _e145 = image_cube_array.sample(sampler_reg, _e3, 0, metal::bias(2.0));
    metal::float4 _e146 = a;
    a = _e146 + _e145;
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
    float _e8 = image_2d_depth.sample_compare(sampler_cmp, tc, 0.5);
    float _e9 = a_1;
    a_1 = _e9 + _e8;
    float _e14 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0u, 0.5);
    float _e15 = a_1;
    a_1 = _e15 + _e14;
    float _e20 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0, 0.5);
    float _e21 = a_1;
    a_1 = _e21 + _e20;
    float _e25 = image_cube_depth.sample_compare(sampler_cmp, tc3_, 0.5);
    float _e26 = a_1;
    a_1 = _e26 + _e25;
    float _e30 = image_2d_depth.sample_compare(sampler_cmp, tc, 0.5);
    float _e31 = a_1;
    a_1 = _e31 + _e30;
    float _e36 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0u, 0.5);
    float _e37 = a_1;
    a_1 = _e37 + _e36;
    float _e42 = image_2d_array_depth.sample_compare(sampler_cmp, tc, 0, 0.5);
    float _e43 = a_1;
    a_1 = _e43 + _e42;
    float _e47 = image_cube_depth.sample_compare(sampler_cmp, tc3_, 0.5);
    float _e48 = a_1;
    a_1 = _e48 + _e47;
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
