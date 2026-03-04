// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};


float test_textureLoad_depth_2d(
    metal::int2 coords,
    int level,
    metal::depth2d<float, metal::access::sample> image_depth_2d
) {
    float _e3 = (uint(level) < image_depth_2d.get_num_mip_levels() && metal::all(metal::uint2(coords) < metal::uint2(image_depth_2d.get_width(level), image_depth_2d.get_height(level))) ? image_depth_2d.read(metal::uint2(coords), level): DefaultConstructible());
    return _e3;
}

float test_textureLoad_depth_2d_array_u(
    metal::int2 coords_1,
    uint index,
    int level_1,
    metal::depth2d_array<float, metal::access::sample> image_depth_2d_array
) {
    float _e4 = (uint(level_1) < image_depth_2d_array.get_num_mip_levels() && uint(index) < image_depth_2d_array.get_array_size() && metal::all(metal::uint2(coords_1) < metal::uint2(image_depth_2d_array.get_width(level_1), image_depth_2d_array.get_height(level_1))) ? image_depth_2d_array.read(metal::uint2(coords_1), index, level_1): DefaultConstructible());
    return _e4;
}

float test_textureLoad_depth_2d_array_s(
    metal::int2 coords_2,
    int index_1,
    int level_2,
    metal::depth2d_array<float, metal::access::sample> image_depth_2d_array
) {
    float _e4 = (uint(level_2) < image_depth_2d_array.get_num_mip_levels() && uint(index_1) < image_depth_2d_array.get_array_size() && metal::all(metal::uint2(coords_2) < metal::uint2(image_depth_2d_array.get_width(level_2), image_depth_2d_array.get_height(level_2))) ? image_depth_2d_array.read(metal::uint2(coords_2), index_1, level_2): DefaultConstructible());
    return _e4;
}

float test_textureLoad_depth_multisampled_2d(
    metal::int2 coords_3,
    int _sample,
    metal::depth2d_ms<float, metal::access::read> image_depth_multisampled_2d
) {
    float _e3 = (uint(_sample) < image_depth_multisampled_2d.get_num_samples() && metal::all(metal::uint2(coords_3) < metal::uint2(image_depth_multisampled_2d.get_width(), image_depth_multisampled_2d.get_height())) ? image_depth_multisampled_2d.read(metal::uint2(coords_3), _sample): DefaultConstructible());
    return _e3;
}

struct fragment_shaderOutput {
    metal::float4 member [[color(0)]];
};
fragment fragment_shaderOutput fragment_shader(
  metal::depth2d<float, metal::access::sample> image_depth_2d [[user(fake0)]]
, metal::depth2d_array<float, metal::access::sample> image_depth_2d_array [[user(fake0)]]
, metal::depth2d_ms<float, metal::access::read> image_depth_multisampled_2d [[user(fake0)]]
) {
    float _e2 = test_textureLoad_depth_2d(metal::int2 {}, 0, image_depth_2d);
    float _e6 = test_textureLoad_depth_2d_array_u(metal::int2 {}, 0u, 0, image_depth_2d_array);
    float _e10 = test_textureLoad_depth_2d_array_s(metal::int2 {}, 0, 0, image_depth_2d_array);
    float _e13 = test_textureLoad_depth_multisampled_2d(metal::int2 {}, 0, image_depth_multisampled_2d);
    return fragment_shaderOutput { metal::float4(0.0, 0.0, 0.0, 0.0) };
}
