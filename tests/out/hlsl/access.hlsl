
RWByteAddressBuffer bar : register(u0);

uint NagaBufferLengthRW(RWByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

float4 foo(uint vi : SV_VertexID) : SV_Position
{
    float foo1 = 0.0;
    int c[5] = {(int)0,(int)0,(int)0,(int)0,(int)0};

    float baz = foo1;
    foo1 = 1.0;
    float4x4 matrix1 = float4x4(asfloat(bar.Load4(0+0)), asfloat(bar.Load4(0+16)), asfloat(bar.Load4(0+32)), asfloat(bar.Load4(0+48)));
    uint2 arr[2] = {asuint(bar.Load2(72+0)), asuint(bar.Load2(72+8))};
    float b = asfloat(bar.Load(0+48+0));
    int a = asint(bar.Load((((NagaBufferLengthRW(bar) - 88) / 4) - 2u)*4+88));
    bar.Store(8+16+0, asuint(1.0));
    {
        float4x4 _value2 = float4x4(float4(0.0.xxxx), float4(1.0.xxxx), float4(2.0.xxxx), float4(3.0.xxxx));
        bar.Store4(0+0, asuint(_value2[0]));
        bar.Store4(0+16, asuint(_value2[1]));
        bar.Store4(0+32, asuint(_value2[2]));
        bar.Store4(0+48, asuint(_value2[3]));
    }
    {
        uint2 _value2[2] = { uint2(0u.xx), uint2(1u.xx) };
        bar.Store2(72+0, asuint(_value2[0]));
        bar.Store2(72+8, asuint(_value2[1]));
    }
    {
        int _result[5]={ a, int(b), 3, 4, 5 };
        for(int _i=0; _i<5; ++_i) c[_i] = _result[_i];
    }
    c[(vi + 1u)] = 42;
    int value = c[vi];
    return mul(float4(int4(value.xxxx)), matrix1);
}

[numthreads(1, 1, 1)]
void atomics()
{
    int tmp = (int)0;

    int value = asint(bar.Load(64));
    int _e6; bar.InterlockedAdd(64, 5, _e6);
    tmp = _e6;
    int _e9; bar.InterlockedAdd(64, -5, _e9);
    tmp = _e9;
    int _e12; bar.InterlockedAnd(64, 5, _e12);
    tmp = _e12;
    int _e15; bar.InterlockedOr(64, 5, _e15);
    tmp = _e15;
    int _e18; bar.InterlockedXor(64, 5, _e18);
    tmp = _e18;
    int _e21; bar.InterlockedMin(64, 5, _e21);
    tmp = _e21;
    int _e24; bar.InterlockedMax(64, 5, _e24);
    tmp = _e24;
    int _e27; bar.InterlockedExchange(64, 5, _e27);
    tmp = _e27;
    bar.Store(64, asuint(value));
    return;
}
