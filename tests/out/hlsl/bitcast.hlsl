[numthreads(1, 1, 1)]
void main()
{
    int2 i2_ = (int2)0;
    int3 i3_ = (int3)0;
    int4 i4_ = (int4)0;
    uint2 u2_ = (uint2)0;
    uint3 u3_ = (uint3)0;
    uint4 u4_ = (uint4)0;
    float2 f2_ = (float2)0;
    float3 f3_ = (float3)0;
    float4 f4_ = (float4)0;

    i2_ = (0).xx;
    i3_ = (0).xxx;
    i4_ = (0).xxxx;
    u2_ = (0u).xx;
    u3_ = (0u).xxx;
    u4_ = (0u).xxxx;
    f2_ = (0.0).xx;
    f3_ = (0.0).xxx;
    f4_ = (0.0).xxxx;
    int2 _expr27 = i2_;
    u2_ = asuint(_expr27);
    int3 _expr29 = i3_;
    u3_ = asuint(_expr29);
    int4 _expr31 = i4_;
    u4_ = asuint(_expr31);
    uint2 _expr33 = u2_;
    i2_ = asint(_expr33);
    uint3 _expr35 = u3_;
    i3_ = asint(_expr35);
    uint4 _expr37 = u4_;
    i4_ = asint(_expr37);
    int2 _expr39 = i2_;
    f2_ = asfloat(_expr39);
    int3 _expr41 = i3_;
    f3_ = asfloat(_expr41);
    int4 _expr43 = i4_;
    f4_ = asfloat(_expr43);
    return;
}
