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

void test_matrix_within_struct_accesses()
{
    int idx = int(1);
    Baz t = ConstructBaz(float3x2((1.0).xx, (2.0).xx, (3.0).xx));

    int _e2 = idx;
    idx = asint(asuint(_e2) - asuint(int(1)));
    float2 _e6 = baz.m_0_;
    float2 _e8 = baz.m_1_;
    float2 _e10 = baz.m_2_;
    float3x2 l0_ = float3x2(_e6, _e8, _e10);
    float2 l1_ = baz.m_0_;
    int _e15 = idx;
    float2 _e18 = baz.m_0_;
    float2 _e20 = baz.m_1_;
    float2 _e22 = baz.m_2_;
    float2 l2_ = float3x2(_e18, _e20, _e22)[min(uint(_e15), 2u)];
    float l3_ = baz.m_0_.y;
    int _e31 = idx;
    float l4_ = baz.m_0_[_e31];
    int _e34 = idx;
    float2 _e37 = baz.m_0_;
    float2 _e39 = baz.m_1_;
    float2 _e41 = baz.m_2_;
    float l5_ = float3x2(_e37, _e39, _e41)[min(uint(_e34), 2u)].y;
    int _e45 = idx;
    int _e46 = idx;
    float2 _e49 = baz.m_0_;
    float2 _e51 = baz.m_1_;
    float2 _e53 = baz.m_2_;
    float l6_ = float3x2(_e49, _e51, _e53)[min(uint(_e45), 2u)][min(uint(_e46), 1u)];
    int _e66 = idx;
    idx = asint(asuint(_e66) + asuint(int(1)));
    t.m = float3x2((6.0).xx, (5.0).xx, (4.0).xx);
    t.m[0] = (9.0).xx;
    int _e82 = idx;
    t.m[min(uint(_e82), 2u)] = (90.0).xx;
    t.m[0].y = 10.0;
    int _e91 = idx;
    t.m[0][min(uint(_e91), 1u)] = 20.0;
    int _e94 = idx;
    t.m[min(uint(_e94), 2u)].y = 30.0;
    int _e98 = idx;
    int _e100 = idx;
    t.m[min(uint(_e98), 2u)][min(uint(_e100), 1u)] = 40.0;
    return;
}

typedef float4x2 ret_Constructarray2_float4x2_[2];
ret_Constructarray2_float4x2_ Constructarray2_float4x2_(float4x2 arg0, float4x2 arg1) {
    float4x2 ret[2] = { arg0, arg1 };
    return ret;
}

typedef float4x2 ret_convert_std140_0_[2];
ret_convert_std140_0_ convert_std140_0_(mat4x2f32_std140_ v[2])
{
    return Constructarray2_float4x2_(float4x2(v[0].c_0_, v[0].c_1_, v[0].c_2_, v[0].c_3_), float4x2(v[1].c_0_, v[1].c_1_, v[1].c_2_, v[1].c_3_));
}

MatCx2InArray ConstructMatCx2InArray(float4x2 arg0[2]) {
    MatCx2InArray ret = (MatCx2InArray)0;
    ret.am = arg0;
    return ret;
}

typedef float4x2 ret_ZeroValuearray2_float4x2_[2];
ret_ZeroValuearray2_float4x2_ ZeroValuearray2_float4x2_() {
    return (float4x2[2])0;
}

void test_matrix_within_array_within_struct_accesses()
{
    int idx_1 = int(1);
    MatCx2InArray t_1 = ConstructMatCx2InArray(ZeroValuearray2_float4x2_());

    int _e2 = idx_1;
    idx_1 = asint(asuint(_e2) - asuint(int(1)));
    mat4x2f32_std140_ _e6[2] = nested_mat_cx2_.am;
    const float4x2 _e132[2] = convert_std140_0_(_e6);
    float2 _e11 = nested_mat_cx2_.am[0].c_0_;
    float2 _e13 = nested_mat_cx2_.am[0].c_1_;
    float2 _e15 = nested_mat_cx2_.am[0].c_2_;
    float2 _e17 = nested_mat_cx2_.am[0].c_3_;
    float4x2 l1_1 = float4x2(_e11, _e13, _e15, _e17);
    float2 l2_1 = nested_mat_cx2_.am[0].c_0_;
    int _e27 = idx_1;
    float2 _e29 = nested_mat_cx2_.am[0].c_0_;
    float2 _e31 = nested_mat_cx2_.am[0].c_1_;
    float2 _e33 = nested_mat_cx2_.am[0].c_2_;
    float2 _e35 = nested_mat_cx2_.am[0].c_3_;
    float2 l3_1 = float4x2(_e29, _e31, _e33, _e35)[min(uint(_e27), 3u)];
    float l4_1 = nested_mat_cx2_.am[0].c_0_.y;
    int _e48 = idx_1;
    float l5_1 = nested_mat_cx2_.am[0].c_0_[_e48];
    int _e54 = idx_1;
    float2 _e56 = nested_mat_cx2_.am[0].c_0_;
    float2 _e58 = nested_mat_cx2_.am[0].c_1_;
    float2 _e60 = nested_mat_cx2_.am[0].c_2_;
    float2 _e62 = nested_mat_cx2_.am[0].c_3_;
    float l6_1 = float4x2(_e56, _e58, _e60, _e62)[min(uint(_e54), 3u)].y;
    int _e69 = idx_1;
    int _e70 = idx_1;
    float2 _e72 = nested_mat_cx2_.am[0].c_0_;
    float2 _e74 = nested_mat_cx2_.am[0].c_1_;
    float2 _e76 = nested_mat_cx2_.am[0].c_2_;
    float2 _e78 = nested_mat_cx2_.am[0].c_3_;
    float l7_ = float4x2(_e72, _e74, _e76, _e78)[min(uint(_e69), 3u)][min(uint(_e70), 1u)];
    int _e85 = idx_1;
    idx_1 = asint(asuint(_e85) + asuint(int(1)));
    t_1.am = ZeroValuearray2_float4x2_();
    t_1.am[0] = float4x2((8.0).xx, (7.0).xx, (6.0).xx, (5.0).xx);
    t_1.am[0][0] = (9.0).xx;
    int _e107 = idx_1;
    t_1.am[0][min(uint(_e107), 3u)] = (90.0).xx;
    t_1.am[0][0].y = 10.0;
    int _e118 = idx_1;
    t_1.am[0][0][min(uint(_e118), 1u)] = 20.0;
    int _e122 = idx_1;
    t_1.am[0][min(uint(_e122), 3u)].y = 30.0;
    int _e127 = idx_1;
    int _e129 = idx_1;
    t_1.am[0][min(uint(_e127), 3u)][min(uint(_e129), 1u)] = 40.0;
    return;
}

float read_from_private(inout float foo_1)
{
    float _e1 = foo_1;
    return _e1;
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

typedef float4 ret_Constructarray2_float4_[2];
ret_Constructarray2_float4_ Constructarray2_float4_(float4 arg0, float4 arg1) {
    float4 ret[2] = { arg0, arg1 };
    return ret;
}

void assign_array_through_ptr_fn(inout float4 foo_2[2])
{
    foo_2 = Constructarray2_float4_((1.0).xxxx, (2.0).xxxx);
    return;
}

void assign_through_ptr()
{
    uint val = 33u;
    float4 arr[2] = Constructarray2_float4_((6.0).xxxx, (7.0).xxxx);

    assign_through_ptr_fn(val);
    assign_array_through_ptr_fn(arr);
    return;
}

uint fetch_arg_ptr_member(inout AssignToMember p_1)
{
    uint _e2 = p_1.x;
    return _e2;
}

void assign_to_arg_ptr_member(inout AssignToMember p_2)
{
    p_2.x = 10u;
    return;
}

uint fetch_arg_ptr_array_element(inout uint p_3[4])
{
    uint _e2 = p_3[1];
    return _e2;
}

void assign_to_arg_ptr_array_element(inout uint p_4[4])
{
    p_4[1] = 10u;
    return;
}

void assign_to_ptr_components()
{
    AssignToMember s1_ = (AssignToMember)0;
    uint a1_[4] = (uint[4])0;

    assign_to_arg_ptr_member(s1_);
    const uint _e1 = fetch_arg_ptr_member(s1_);
    assign_to_arg_ptr_array_element(a1_);
    const uint _e3 = fetch_arg_ptr_array_element(a1_);
    return;
}

typedef bool ret_Constructarray1_bool_[1];
ret_Constructarray1_bool_ Constructarray1_bool_(bool arg0) {
    bool ret[1] = { arg0 };
    return ret;
}

bool index_ptr(bool value)
{
    bool a_1[1] = (bool[1])0;

    a_1 = Constructarray1_bool_(value);
    bool _e4 = a_1[0];
    return _e4;
}

S ConstructS(int arg0) {
    S ret = (S)0;
    ret.m = arg0;
    return ret;
}

int member_ptr()
{
    S s = ConstructS(int(42));

    int _e4 = s.m;
    return _e4;
}

Outer ZeroValueOuter() {
    return (Outer)0;
}

int let_members_of_members()
{
    Inner inner_1 = ZeroValueOuter().om_nom_nom;
    int delishus_1 = inner_1.delicious;
    if ((ZeroValueOuter().thing != uint(delishus_1))) {
    }
    return ZeroValueOuter().om_nom_nom.delicious;
}

int var_members_of_members()
{
    Outer thing = ZeroValueOuter();
    Inner inner = (Inner)0;
    int delishus = (int)0;

    Inner _e3 = thing.om_nom_nom;
    inner = _e3;
    int _e6 = inner.delicious;
    delishus = _e6;
    uint _e9 = thing.thing;
    int _e10 = delishus;
    if ((_e9 != uint(_e10))) {
    }
    int _e15 = thing.om_nom_nom.delicious;
    return _e15;
}

typedef int ret_Constructarray5_int_[5];
ret_Constructarray5_int_ Constructarray5_int_(int arg0, int arg1, int arg2, int arg3, int arg4) {
    int ret[5] = { arg0, arg1, arg2, arg3, arg4 };
    return ret;
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

    float baz_1 = foo;
    foo = 1.0;
    GlobalConst phony = msl_padding_global_const;
    test_matrix_within_struct_accesses();
    test_matrix_within_array_within_struct_accesses();
    float4x3 _matrix = float4x3(asfloat(bar.Load3(0+0)), asfloat(bar.Load3(0+16)), asfloat(bar.Load3(0+32)), asfloat(bar.Load3(0+48)));
    uint2 arr_1[2] = Constructarray2_uint2_(asuint(bar.Load2(144+0)), asuint(bar.Load2(144+8)));
    float b = asfloat(bar.Load(0+3u*16+0));
    int a_2 = asint(bar.Load(0+(((NagaBufferLengthRW(bar) - 160) / 8) - 2u)*8+160));
    int2 c = asint(qux.Load2(0));
    const float _e35 = read_from_private(foo);
    c2_ = Constructarray5_int_(a_2, naga_f2i32(b), int(3), int(4), int(5));
    c2_[min(uint((vi + 1u)), 4u)] = int(42);
    int value_1 = c2_[min(uint(vi), 4u)];
    const float _e49 = test_arr_as_arg(ZeroValuearray5_array10_float__());
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

[numthreads(1, 1, 1)]
void foo_compute()
{
    assign_through_ptr();
    assign_to_ptr_components();
    const bool _e1 = index_ptr(true);
    const int _e2 = member_ptr();
    const int _e3 = let_members_of_members();
    const int _e4 = var_members_of_members();
    return;
}
