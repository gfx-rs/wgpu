
struct GlobalConst {
    uint a;
    int _pad1_0;
    int _pad1_1;
    int _pad1_2;
    uint3 b;
    int c;
};

struct AlignedWrapper {
    int value;
    int _end_pad_0;
};

struct Baz {
    float2 m_0; float2 m_1; float2 m_2;
};

GlobalConst ConstructGlobalConst(uint arg0, uint3 arg1, int arg2) {
    GlobalConst ret = (GlobalConst)0;
    ret.a = arg0;
    ret.b = arg1;
    ret.c = arg2;
    return ret;
}

typedef float ret_Constructarray10_float_[10];
ret_Constructarray10_float_ Constructarray10_float_(float arg0, float arg1, float arg2, float arg3, float arg4, float arg5, float arg6, float arg7, float arg8, float arg9) {
    float ret[10] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9 };
    return ret;
}

typedef float ret_Constructarray5_array10_float__[5][10];
ret_Constructarray5_array10_float__ Constructarray5_array10_float__(float arg0[10], float arg1[10], float arg2[10], float arg3[10], float arg4[10]) {
    float ret[5][10] = { arg0, arg1, arg2, arg3, arg4 };
    return ret;
}

static GlobalConst global_const = ConstructGlobalConst(0u, uint3(0u, 0u, 0u), 0);
RWByteAddressBuffer bar : register(u0);
cbuffer baz : register(b1) { Baz baz; }
RWByteAddressBuffer qux : register(u2);
groupshared uint val;

float3x2 GetMatmOnBaz(Baz obj) {
    return float3x2(obj.m_0, obj.m_1, obj.m_2);
}

void SetMatmOnBaz(Baz obj, float3x2 mat) {
    obj.m_0 = mat[0];
    obj.m_1 = mat[1];
    obj.m_2 = mat[2];
}

void SetMatVecmOnBaz(Baz obj, float2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0 = vec; break; }
    case 1: { obj.m_1 = vec; break; }
    case 2: { obj.m_2 = vec; break; }
    }
}

void SetMatScalarmOnBaz(Baz obj, float scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.m_0[vec_idx] = scalar; break; }
    case 1: { obj.m_1[vec_idx] = scalar; break; }
    case 2: { obj.m_2[vec_idx] = scalar; break; }
    }
}

Baz ConstructBaz(float3x2 arg0) {
    Baz ret = (Baz)0;
    ret.m_0 = arg0[0];
    ret.m_1 = arg0[1];
    ret.m_2 = arg0[2];
    return ret;
}

void test_matrix_within_struct_accesses()
{
    int idx = 1;
    Baz t = (Baz)0;

    int _expr6 = idx;
    idx = (_expr6 - 1);
    float3x2 unnamed = GetMatmOnBaz(baz);
    float2 unnamed_1 = GetMatmOnBaz(baz)[0];
    int _expr16 = idx;
    float2 unnamed_2 = GetMatmOnBaz(baz)[_expr16];
    float unnamed_3 = GetMatmOnBaz(baz)[0][1];
    int _expr28 = idx;
    float unnamed_4 = GetMatmOnBaz(baz)[0][_expr28];
    int _expr32 = idx;
    float unnamed_5 = GetMatmOnBaz(baz)[_expr32][1];
    int _expr38 = idx;
    int _expr40 = idx;
    float unnamed_6 = GetMatmOnBaz(baz)[_expr38][_expr40];
    t = ConstructBaz(float3x2((1.0).xx, (2.0).xx, (3.0).xx));
    int _expr52 = idx;
    idx = (_expr52 + 1);
    SetMatmOnBaz(t, float3x2((6.0).xx, (5.0).xx, (4.0).xx));
    t.m_0 = (9.0).xx;
    int _expr69 = idx;
    SetMatVecmOnBaz(t, (90.0).xx, _expr69);
    t.m_0[1] = 10.0;
    int _expr82 = idx;
    t.m_0[_expr82] = 20.0;
    int _expr86 = idx;
    SetMatScalarmOnBaz(t, 30.0, _expr86, 1);
    int _expr92 = idx;
    int _expr94 = idx;
    SetMatScalarmOnBaz(t, 40.0, _expr92, _expr94);
    return;
}

float read_from_private(inout float foo_1)
{
    float _expr5 = foo_1;
    return _expr5;
}

float test_arr_as_arg(float a[5][10])
{
    return a[4][9];
}

void assign_through_ptr_fn(inout uint p)
{
    p = 42u;
    return;
}

uint NagaBufferLengthRW(RWByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

typedef int ret_Constructarray5_int_[5];
ret_Constructarray5_int_ Constructarray5_int_(int arg0, int arg1, int arg2, int arg3, int arg4) {
    int ret[5] = { arg0, arg1, arg2, arg3, arg4 };
    return ret;
}

float4 foo_vert(uint vi : SV_VertexID) : SV_Position
{
    float foo = 0.0;
    int c[5] = {(int)0,(int)0,(int)0,(int)0,(int)0};

    float baz_1 = foo;
    foo = 1.0;
    test_matrix_within_struct_accesses();
    float4x3 _matrix = float4x3(asfloat(bar.Load3(0+0)), asfloat(bar.Load3(0+16)), asfloat(bar.Load3(0+32)), asfloat(bar.Load3(0+48)));
    uint2 arr[2] = {asuint(bar.Load2(104+0)), asuint(bar.Load2(104+8))};
    float b = asfloat(bar.Load(0+48+0));
    int a_1 = asint(bar.Load(0+(((NagaBufferLengthRW(bar) - 120) / 8) - 2u)*8+120));
    int2 c_1 = asint(qux.Load2(0));
    const float _e31 = read_from_private(foo);
    {
        int _result[5]=Constructarray5_int_(a_1, int(b), 3, 4, 5);
        for(int _i=0; _i<5; ++_i) c[_i] = _result[_i];
    }
    c[(vi + 1u)] = 42;
    int value = c[vi];
    const float _e45 = test_arr_as_arg(Constructarray5_array10_float__(Constructarray10_float_(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), Constructarray10_float_(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), Constructarray10_float_(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), Constructarray10_float_(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), Constructarray10_float_(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
    return float4(mul(float4((value).xxxx), _matrix), 2.0);
}

typedef uint2 ret_Constructarray2_uint2_[2];
ret_Constructarray2_uint2_ Constructarray2_uint2_(uint2 arg0, uint2 arg1) {
    uint2 ret[2] = { arg0, arg1 };
    return ret;
}

float4 foo_frag() : SV_Target0
{
    bar.Store(8+16+0, asuint(1.0));
    {
        float4x3 _value2 = float4x3((0.0).xxx, (1.0).xxx, (2.0).xxx, (3.0).xxx);
        bar.Store3(0+0, asuint(_value2[0]));
        bar.Store3(0+16, asuint(_value2[1]));
        bar.Store3(0+32, asuint(_value2[2]));
        bar.Store3(0+48, asuint(_value2[3]));
    }
    {
        uint2 _value2[2] = Constructarray2_uint2_((0u).xx, (1u).xx);
        bar.Store2(104+0, asuint(_value2[0]));
        bar.Store2(104+8, asuint(_value2[1]));
    }
    bar.Store(0+8+120, asuint(1));
    qux.Store2(0, asuint(int2(0, 0)));
    return (0.0).xxxx;
}

[numthreads(1, 1, 1)]
void atomics()
{
    int tmp = (int)0;

    int value_1 = asint(bar.Load(96));
    int _e9; bar.InterlockedAdd(96, 5, _e9);
    tmp = _e9;
    int _e12; bar.InterlockedAdd(96, -5, _e12);
    tmp = _e12;
    int _e15; bar.InterlockedAnd(96, 5, _e15);
    tmp = _e15;
    int _e18; bar.InterlockedOr(96, 5, _e18);
    tmp = _e18;
    int _e21; bar.InterlockedXor(96, 5, _e21);
    tmp = _e21;
    int _e24; bar.InterlockedMin(96, 5, _e24);
    tmp = _e24;
    int _e27; bar.InterlockedMax(96, 5, _e27);
    tmp = _e27;
    int _e30; bar.InterlockedExchange(96, 5, _e30);
    tmp = _e30;
    bar.Store(96, asuint(value_1));
    return;
}

[numthreads(1, 1, 1)]
void assign_through_ptr()
{
    assign_through_ptr_fn(val);
    return;
}
