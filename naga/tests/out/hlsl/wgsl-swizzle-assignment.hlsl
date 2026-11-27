static uint4 v_1[1] = (uint4[1])0;
RWByteAddressBuffer buf : register(u0);
groupshared float2x4 w;

uint update_and_index()
{
    uint4 _e2 = v_1[0];
    v_1[0] = (_e2 + uint4(2u, 4u, 8u, 10u));
    return 0u;
}

uint2 update_and_rhs()
{
    v_1[0].x = 900u;
    v_1[0].z = 100u;
    return uint2(55u, 66u);
}

uint4 eval_order()
{
    v_1[0] = uint4(0u, 0u, 0u, 0u);
    const uint _e8 = update_and_index();
    const uint2 _e10 = update_and_rhs();
    uint4 _e11 = v_1[min(uint(_e8), 0u)];
    v_1[min(uint(_e8), 0u)] = uint4(_e10.y, _e10.x, _e11.z, _e11.w);
    uint4 assigned = v_1[0];
    v_1[0] = uint4(0u, 0u, 0u, 0u);
    const uint _e28 = update_and_index();
    uint4 _e30 = v_1[min(uint(_e28), 0u)];
    const uint2 _e32 = update_and_rhs();
    uint2 _e33 = (_e30.yx + _e32);
    uint4 _e34 = v_1[min(uint(_e28), 0u)];
    v_1[min(uint(_e28), 0u)] = uint4(_e33.y, _e33.x, _e34.z, _e34.w);
    uint4 _e42 = v_1[0];
    return (assigned + _e42);
}

uint4 swizzle_views(int i)
{
    uint4 v = (uint4)0;

    uint4 _e4 = v;
    v = uint4(_e4.x, 1u, 2u, _e4.w);
    uint4 _e8 = v;
    uint3 u = _e8.wyz;
    v = uint4(1u, 2u, 3u, 4u);
    v = uint4(4u, 3u, 2u, 1u);
    uint4 _e22 = v;
    v = uint4(_e22.x, 100u, 99u, _e22.w);
    v.z = 99u;
    uint _e30 = v.z;
    v.z = (_e30 + 1u);
    v.y = 50u;
    v[min(uint(((i == int(1)) ? int(0) : int(3))), 3u)] = 7u;
    v[min(uint(i), 3u)] = 8u;
    uint4 _e43 = v;
    uint2 _e46 = (_e43.xz * 3u);
    uint4 _e47 = v;
    v = uint4(_e46.x, _e47.y, _e46.y, _e47.w);
    uint4 _e53 = v;
    uint2 _e58 = (_e53.yw << uint2(1u, 2u));
    uint4 _e59 = v;
    v = uint4(_e59.x, _e58.x, _e59.z, _e58.y);
    uint4 _e65 = v;
    return (_e65 + u.xyzx);
}

void via_pointer(inout uint4 p)
{
    uint4 _e3 = p;
    p = uint4(_e3.x, 1u, 2u, _e3.w);
    uint4 _e7 = p;
    uint3 u_1 = _e7.wyz;
    uint4 _e11 = p;
    p = uint4(_e11.x, 100u, 99u, _e11.w);
    p.y = 50u;
    uint2 _e17 = u_1.xy;
    uint4 _e18 = p;
    p = uint4(_e17.x, _e18.y, _e18.z, _e17.y);
    return;
}

uint4 ZeroValueuint4() {
    return (uint4)0;
}

[numthreads(1, 1, 1)]
void main(uint index : SV_GroupIndex)
{
    if (index == 0) {
        w = (float2x4)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint4 a = ZeroValueuint4();

    via_pointer(a);
    float3 _e7 = asfloat(buf.Load3(index*16));
    buf.Store3(index*16, asuint(float3(2.0, _e7.y, 1.0)));
    float3 _e12 = asfloat(buf.Load3(index*16));
    float2 _e16 = (_e12.yz - (1.0).xx);
    float3 _e17 = asfloat(buf.Load3(index*16));
    buf.Store3(index*16, asuint(float3(_e17.x, _e16.x, _e16.y)));
    float4 _e26 = w[1];
    w[1] = float4(_e26.x, 4.0, _e26.z, 3.0);
    const uint4 _e30 = eval_order();
    const uint4 _e32 = swizzle_views(int(1));
    uint4 _e34 = a;
    uint4 r = ((_e30 + _e32) + _e34);
    float4 _e42 = w[1];
    buf.Store3(0, asuint((float3(r.xyz) + _e42.xyz)));
    return;
}
