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
    row_major float3x2 m;
};

struct MatCx2InArray {
    row_major float4x2 am[2];
};

struct AssignToMember {
    uint x;
};

struct S {
    int m;
};

struct Inner {
    int delicious;
};

struct Outer {
    Inner om_nom_nom;
    uint thing;
};

struct Baz_std140_ {
    float2 m_0_;
    float2 m_1_;
    float2 m_2_;
};

struct mat4x2f32_std140_ {
    float2 c_0_;
    float2 c_1_;
    float2 c_2_;
    float2 c_3_;
};

struct MatCx2InArray_std140_ {
    mat4x2f32_std140_ am[2];
};

GlobalConst ConstructGlobalConst(uint arg0, uint3 arg1, int arg2) {
    GlobalConst ret = (GlobalConst)0;
    ret.a = arg0;
    ret.b = arg1;
    ret.c = arg2;
    return ret;
}

static GlobalConst msl_padding_global_const = ConstructGlobalConst(0u, uint3(0u, 0u, 0u), int(0));
RWByteAddressBuffer bar : register(u0);
cbuffer baz : register(b1) { Baz_std140_ baz; }
RWByteAddressBuffer qux : register(u2);
cbuffer nested_mat_cx2_ : register(b3) { MatCx2InArray_std140_ nested_mat_cx2_; }

Baz ConstructBaz(float3x2 arg0) {
    Baz ret = (Baz)0;
    ret.m = arg0;
    return ret;
}

MatCx2InArray ConstructMatCx2InArray(float4x2 arg0[2]) {
    MatCx2InArray ret = (MatCx2InArray)0;
    ret.am = arg0;
    return ret;
}

typedef float4x2 ret_Constructarray2_float4x2_[2];
ret_Constructarray2_float4x2_ Constructarray2_float4x2_(float4x2 arg0, float4x2 arg1) {
    float4x2 ret[2] = { arg0, arg1 };
    return ret;
}

typedef int ret_Constructarray5_int_[5];
ret_Constructarray5_int_ Constructarray5_int_(int arg0, int arg1, int arg2, int arg3, int arg4) {
    int ret[5] = { arg0, arg1, arg2, arg3, arg4 };
    return ret;
}

typedef float4x2 ret_ZeroValuearray2_float4x2_[2];
ret_ZeroValuearray2_float4x2_ ZeroValuearray2_float4x2_() {
    return (float4x2[2])0;
}

typedef float ret_ZeroValuearray5_array10_float__[5][10];
ret_ZeroValuearray5_array10_float__ ZeroValuearray5_array10_float__() {
    return (float[5][10])0;
}

int naga_f2i32(float value) {
    return int(clamp(value, -2147483600.0, 2147483500.0));
}

typedef uint2 ret_Constructarray2_uint2_[2];
ret_Constructarray2_uint2_ Constructarray2_uint2_(uint2 arg0, uint2 arg1) {
    uint2 ret[2] = { arg0, arg1 };
    return ret;
}

uint NagaBufferLengthRW(RWByteAddressBuffer buffer)
{
    uint ret;
    buffer.GetDimensions(ret);
    return ret;
}

float4 foo_vert(uint vi : SV_VertexID) : SV_Position
{
    float foo = 0.0;
    int c2_[5] = (int[5])0;
    int idx = (int)0;
    Baz t = (Baz)0;
    int idx_1 = (int)0;
    MatCx2InArray t_1 = (MatCx2InArray)0;
    float local = (float)0;
    float a[5][10] = (float[5][10])0;
    float local_1 = (float)0;
    mat4x2f32_std140_ v[2] = (mat4x2f32_std140_[2])0;
    float4x2 local_2[2] = (float4x2[2])0;

    float baz_1 = foo;
    foo = 1.0;
    GlobalConst phony = msl_padding_global_const;
    idx = int(1);
    t = ConstructBaz(float3x2((1.0).xx, (2.0).xx, (3.0).xx));
    int _e14 = idx;
    idx = asint(asuint(_e14) - asuint(int(1)));
    int _e17 = idx;
    idx = asint(asuint(_e17) + asuint(int(1)));
    t.m = float3x2((6.0).xx, (5.0).xx, (4.0).xx);
    t.m[0] = (9.0).xx;
    int _e33 = idx;
    t.m[min(uint(_e33), 2u)] = (90.0).xx;
    t.m[0].y = 10.0;
    int _e42 = idx;
    t.m[0][min(uint(_e42), 1u)] = 20.0;
    int _e45 = idx;
    t.m[min(uint(_e45), 2u)].y = 30.0;
    int _e49 = idx;
    int _e51 = idx;
    t.m[min(uint(_e49), 2u)][min(uint(_e51), 1u)] = 40.0;
    idx_1 = int(1);
    t_1 = ConstructMatCx2InArray(ZeroValuearray2_float4x2_());
    int _e57 = idx_1;
    idx_1 = asint(asuint(_e57) - asuint(int(1)));
    mat4x2f32_std140_ _e61[2] = nested_mat_cx2_.am;
    v = _e61;
    mat4x2f32_std140_ _e63[2] = v;
    local_2 = Constructarray2_float4x2_(float4x2(_e63[0].c_0_, _e63[0].c_1_, _e63[0].c_2_, _e63[0].c_3_), float4x2(_e63[1].c_0_, _e63[1].c_1_, _e63[1].c_2_, _e63[1].c_3_));
    int _e78 = idx_1;
    idx_1 = asint(asuint(_e78) + asuint(int(1)));
    t_1.am = ZeroValuearray2_float4x2_();
    t_1.am[0] = float4x2((8.0).xx, (7.0).xx, (6.0).xx, (5.0).xx);
    t_1.am[0][0] = (9.0).xx;
    int _e100 = idx_1;
    t_1.am[0][min(uint(_e100), 3u)] = (90.0).xx;
    t_1.am[0][0].y = 10.0;
    int _e111 = idx_1;
    t_1.am[0][0][min(uint(_e111), 1u)] = 20.0;
    int _e115 = idx_1;
    t_1.am[0][min(uint(_e115), 3u)].y = 30.0;
    int _e120 = idx_1;
    int _e122 = idx_1;
    t_1.am[0][min(uint(_e120), 3u)][min(uint(_e122), 1u)] = 40.0;
    float4x3 _matrix = float4x3(asfloat(bar.Load3(0+0)), asfloat(bar.Load3(0+16)), asfloat(bar.Load3(0+32)), asfloat(bar.Load3(0+48)));
    uint2 arr_1[2] = Constructarray2_uint2_(asuint(bar.Load2(144+0)), asuint(bar.Load2(144+8)));
    float b = asfloat(bar.Load(0+3u*16+0));
    int a_2 = asint(bar.Load(0+(((NagaBufferLengthRW(bar) - 160) / 8) - 2u)*8+160));
    int2 c = asint(qux.Load2(0));
    float _e152 = foo;
    local = _e152;
    float foo_value = local;
    c2_ = Constructarray5_int_(a_2, naga_f2i32(b), int(3), int(4), int(5));
    c2_[min(uint((vi + 1u)), 4u)] = int(42);
    int value_1 = c2_[min(uint(vi), 4u)];
    a = ZeroValuearray5_array10_float__();
    float _e168[5][10] = a;
    local_1 = _e168[4][9];
    return float4(mul(float4((value_1).xxxx), _matrix), 2.0);
}

int2 ZeroValueint2() {
    return (int2)0;
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
        bar.Store2(144+0, asuint(_value2[0]));
        bar.Store2(144+8, asuint(_value2[1]));
    }
    bar.Store(0+8+160, asuint(int(1)));
    qux.Store2(0, asuint(ZeroValueint2()));
    return (0.0).xxxx;
}

typedef float4 ret_Constructarray2_float4_[2];
ret_Constructarray2_float4_ Constructarray2_float4_(float4 arg0, float4 arg1) {
    float4 ret[2] = { arg0, arg1 };
    return ret;
}

typedef bool ret_Constructarray1_bool_[1];
ret_Constructarray1_bool_ Constructarray1_bool_(bool arg0) {
    bool ret[1] = { arg0 };
    return ret;
}

S ConstructS(int arg0) {
    S ret = (S)0;
    ret.m = arg0;
    return ret;
}

AssignToMember ZeroValueAssignToMember() {
    return (AssignToMember)0;
}

typedef uint ret_ZeroValuearray4_uint_[4];
ret_ZeroValuearray4_uint_ ZeroValuearray4_uint_() {
    return (uint[4])0;
}

uint ZeroValueuint() {
    return (uint)0;
}

typedef bool ret_ZeroValuearray1_bool_[1];
ret_ZeroValuearray1_bool_ ZeroValuearray1_bool_() {
    return (bool[1])0;
}

Outer ZeroValueOuter() {
    return (Outer)0;
}

Inner ZeroValueInner() {
    return (Inner)0;
}

int ZeroValueint() {
    return (int)0;
}

[numthreads(1, 1, 1)]
void foo_compute()
{
    uint val = (uint)0;
    float4 arr[2] = (float4[2])0;
    AssignToMember s1_ = (AssignToMember)0;
    uint a1_[4] = (uint[4])0;
    uint local_3 = (uint)0;
    uint local_4 = (uint)0;
    bool value = (bool)0;
    bool a_1[1] = (bool[1])0;
    bool local_5 = (bool)0;
    S s = (S)0;
    int local_6 = (int)0;
    int local_7 = (int)0;
    Outer thing = (Outer)0;
    Inner inner = (Inner)0;
    int delishus = (int)0;
    int local_8 = (int)0;

    val = 33u;
    arr = Constructarray2_float4_((6.0).xxxx, (7.0).xxxx);
    val = 42u;
    arr = Constructarray2_float4_((1.0).xxxx, (2.0).xxxx);
    s1_ = ZeroValueAssignToMember();
    a1_ = ZeroValuearray4_uint_();
    local_3 = ZeroValueuint();
    local_4 = ZeroValueuint();
    s1_.x = 10u;
    uint _e22 = s1_.x;
    local_3 = _e22;
    a1_[1] = 10u;
    uint _e27 = a1_[1];
    local_4 = _e27;
    value = true;
    bool _e38 = value;
    a_1 = ZeroValuearray1_bool_();
    a_1 = Constructarray1_bool_(_e38);
    bool _e42 = a_1[0];
    local_5 = _e42;
    s = ConstructS(int(42));
    int _e49 = s.m;
    local_6 = _e49;
    if ((ZeroValueOuter().thing != uint(ZeroValueOuter().om_nom_nom.delicious))) {
    }
    local_7 = ZeroValueOuter().om_nom_nom.delicious;
    thing = ZeroValueOuter();
    inner = ZeroValueInner();
    delishus = ZeroValueint();
    Inner _e65 = thing.om_nom_nom;
    inner = _e65;
    int _e68 = inner.delicious;
    delishus = _e68;
    uint _e71 = thing.thing;
    int _e72 = delishus;
    if ((_e71 != uint(_e72))) {
    }
    int _e77 = thing.om_nom_nom.delicious;
    local_8 = _e77;
    return;
}
