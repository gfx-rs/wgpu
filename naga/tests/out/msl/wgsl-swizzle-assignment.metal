// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size1;
};

struct type_1 {
    metal::uint4 inner[1];
};
typedef metal::float3 type_7[1];

uint update_and_index(
    thread type_1& v_1
) {
    metal::uint4 _e2 = v_1.inner[0];
    v_1.inner[0] = _e2 + metal::uint4(2u, 4u, 8u, 10u);
    return 0u;
}

metal::uint2 update_and_rhs(
    thread type_1& v_1
) {
    v_1.inner[0].x = 900u;
    v_1.inner[0].z = 100u;
    return metal::uint2(55u, 66u);
}

metal::uint4 eval_order(
    thread type_1& v_1
) {
    v_1.inner[0] = metal::uint4(0u, 0u, 0u, 0u);
    uint _e8 = update_and_index(v_1);
    metal::uint2 _e10 = update_and_rhs(v_1);
    metal::uint4 _e11 = v_1.inner[_e8];
    v_1.inner[_e8] = metal::uint4(_e10.y, _e10.x, _e11.z, _e11.w);
    metal::uint4 assigned = v_1.inner[0];
    v_1.inner[0] = metal::uint4(0u, 0u, 0u, 0u);
    uint _e28 = update_and_index(v_1);
    metal::uint4 _e30 = v_1.inner[_e28];
    metal::uint2 _e32 = update_and_rhs(v_1);
    metal::uint2 _e33 = _e30.yx + _e32;
    metal::uint4 _e34 = v_1.inner[_e28];
    v_1.inner[_e28] = metal::uint4(_e33.y, _e33.x, _e34.z, _e34.w);
    metal::uint4 _e42 = v_1.inner[0];
    return assigned + _e42;
}

metal::uint4 swizzle_views(
    int i
) {
    metal::uint4 v = {};
    metal::uint4 _e4 = v;
    v = metal::uint4(_e4.x, 1u, 2u, _e4.w);
    metal::uint4 _e8 = v;
    metal::uint3 u = _e8.wyz;
    v = metal::uint4(1u, 2u, 3u, 4u);
    v = metal::uint4(4u, 3u, 2u, 1u);
    metal::uint4 _e22 = v;
    v = metal::uint4(_e22.x, 100u, 99u, _e22.w);
    v.z = 99u;
    uint _e30 = v.z;
    v.z = _e30 + 1u;
    v.y = 50u;
    v[(i == 1) ? 0 : 3] = 7u;
    v[i] = 8u;
    metal::uint4 _e43 = v;
    metal::uint2 _e46 = _e43.xz * 3u;
    metal::uint4 _e47 = v;
    v = metal::uint4(_e46.x, _e47.y, _e46.y, _e47.w);
    metal::uint4 _e53 = v;
    metal::uint2 _e58 = _e53.yw << metal::uint2(1u, 2u);
    metal::uint4 _e59 = v;
    v = metal::uint4(_e59.x, _e58.x, _e59.z, _e58.y);
    metal::uint4 _e65 = v;
    return _e65 + u.xyzx;
}

void via_pointer(
    thread metal::uint4& p
) {
    metal::uint4 _e3 = p;
    p = metal::uint4(_e3.x, 1u, 2u, _e3.w);
    metal::uint4 _e7 = p;
    metal::uint3 u_1 = _e7.wyz;
    metal::uint4 _e11 = p;
    p = metal::uint4(_e11.x, 100u, 99u, _e11.w);
    p.y = 50u;
    metal::uint2 _e17 = u_1.xy;
    metal::uint4 _e18 = p;
    p = metal::uint4(_e17.x, _e18.y, _e18.z, _e17.y);
    return;
}

struct main_Input {
};
kernel void main_(
  uint index [[thread_index_in_threadgroup]]
, device type_7& buf [[user(fake0)]]
, threadgroup metal::float2x4& w
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    type_1 v_1 = {};
    if (index == 0u) {
        w = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    metal::uint4 a = metal::uint4 {};
    via_pointer(a);
    metal::float3 _e7 = buf[index];
    buf[index] = metal::float3(2.0, _e7.y, 1.0);
    metal::float3 _e12 = buf[index];
    metal::float2 _e16 = _e12.yz - metal::float2(1.0);
    metal::float3 _e17 = buf[index];
    buf[index] = metal::float3(_e17.x, _e16.x, _e16.y);
    metal::float4 _e26 = w[1];
    w[1] = metal::float4(_e26.x, 4.0, _e26.z, 3.0);
    metal::uint4 _e30 = eval_order(v_1);
    metal::uint4 _e32 = swizzle_views(1);
    metal::uint4 _e34 = a;
    metal::uint4 r = (_e30 + _e32) + _e34;
    metal::float4 _e42 = w[1];
    buf[0] = static_cast<metal::float3>(r.xyz) + _e42.xyz;
    return;
}
