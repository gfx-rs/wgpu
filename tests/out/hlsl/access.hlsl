
RWByteAddressBuffer bar : register(u0);

struct VertexInput_foo {
    uint vi1 : SV_VertexID;
};

uint NagaBufferLengthRW(RWByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

float4 foo(VertexInput_foo vertexinput_foo) : SV_Position
{
    float foo1 = 0.0;
    int c[5] = {(int)0,(int)0,(int)0,(int)0,(int)0};

    float baz = foo1;
    foo1 = 1.0;
    float4x4 matrix1 = float4x4(asfloat(bar.Load4(0+0)), asfloat(bar.Load4(0+16)), asfloat(bar.Load4(0+32)), asfloat(bar.Load4(0+48)));
    uint2 arr[2] = {asuint(bar.Load2(4+0)), asuint(bar.Load2(4+8))};
    float4 _expr13 = asfloat(bar.Load4(12+0));
    float b = _expr13.x;
    int a = asint(bar.Load((((NagaBufferLengthRW(bar) - 80) / 4) - 2u)*4+8));
    {
        int _result[5]={ a, int(b), 3, 4, 5 };
        for(int _i=0; _i<5; ++_i) c[_i] = _result[_i];
    }
    c[(vertexinput_foo.vi1 + 1u)] = 42;
    int value = c[vertexinput_foo.vi1];
    return mul(matrix1, float4(int4(value.xxxx)));
}
