// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_2 {
    metal::float4 inner[10];
};
struct InStorage {
    type_2 a;
};
struct type_3 {
    metal::float4 inner[20];
};
struct InUniform {
    type_3 a;
};
struct type_5 {
    float inner[30];
};
struct type_6 {
    float inner[40];
};
struct type_9 {
    metal::float4 inner[2];
};

metal::float4 mock_function(
    metal::int2 c,
    int i,
    int l,
    device InStorage const& in_storage,
    constant InUniform& in_uniform,
    metal::texture2d_array<float, metal::access::sample> image_2d_array,
    threadgroup type_5* in_workgroup,
    thread type_6& in_private
) {
    type_9 in_function = type_9 {{metal::float4(0.707, 0.0, 0.0, 1.0), metal::float4(0.0, 0.707, 0.0, 1.0)}};
    metal::float4 _e18 = in_storage.a.inner[i];
    metal::float4 _e22 = in_uniform.a.inner[i];
    metal::float4 _e25 = image_2d_array.read(metal::uint2(c), i, l);
    float _e29 = in_workgroup->inner[i];
    float _e34 = in_private.inner[i];
    metal::float4 _e38 = in_function.inner[i];
    return ((((_e18 + _e22) + _e25) + metal::float4(_e29)) + metal::float4(_e34)) + _e38;
}

kernel void main_(
  metal::uint3 __local_invocation_id [[thread_position_in_threadgroup]]
, device InStorage const& in_storage [[user(fake0)]]
, constant InUniform& in_uniform [[user(fake0)]]
, metal::texture2d_array<float, metal::access::sample> image_2d_array [[user(fake0)]]
, threadgroup type_5* in_workgroup
) {
    if (metal::all(__local_invocation_id == metal::uint3(0u))) {
        in_workgroup = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    type_6 in_private = {};
    metal::float4 _e5 = mock_function(metal::int2(1, 2), 3, 4, in_storage, in_uniform, image_2d_array, in_workgroup, in_private);
    return;
}
