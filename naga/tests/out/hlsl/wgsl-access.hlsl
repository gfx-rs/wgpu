typedef struct { float2 _0; float2 _1; } __mat2x2_f32;
float2 __get_col_of_mat2x2_f32(__mat2x2_f32 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat2x2_f32(__mat2x2_f32 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    }
}
void __set_el_of_mat2x2_f32(__mat2x2_f32 mat, uint idx, uint vec_idx, float value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    }
}

typedef struct { float2 _0; float2 _1; float2 _2; float2 _3; } __mat4x2_f32;
float2 __get_col_of_mat4x2_f32(__mat4x2_f32 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    case 2: { return mat._2; }
    case 3: { return mat._3; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat4x2_f32(__mat4x2_f32 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    case 2: { mat._2 = value; break; }
    case 3: { mat._3 = value; break; }
    }
}
void __set_el_of_mat4x2_f32(__mat4x2_f32 mat, uint idx, uint vec_idx, float value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    case 2: { mat._2[vec_idx] = value; break; }
    case 3: { mat._3[vec_idx] = value; break; }
    }
}

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

struct MatCx2InArray {
    __mat4x2_f32 am[2];
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

GlobalConst ConstructGlobalConst(uint arg0, uint3 arg1, int arg2) {
    GlobalConst ret = (GlobalConst)0;
    ret.a = arg0;
    ret.b = arg1;
    ret.c = arg2;
    return ret;
}

static GlobalConst msl_padding_global_const = ConstructGlobalConst(0u, uint3(0u, 0u, 0u), int(0));
RWByteAddressBuffer bar : register(u0);
cbuffer baz : register(b1) { Baz baz; }
RWByteAddressBuffer qux : register(u2);
cbuffer nested_mat_cx2_ : register(b3) { MatCx2InArray nested_mat_cx2_; }

Baz ConstructBaz(float3x2 arg0) {
    Baz ret = (Baz)0;
    ret.m_0 = arg0[0];
    ret.m_1 = arg0[1];
    ret.m_2 = arg0[2];
    return ret;
}

MatCx2InArray ConstructMatCx2InArray(float4x2 arg0[2]) {
    MatCx2InArray ret = (MatCx2InArray)0;
    ret.am = (__mat4x2_f32[2])arg0;
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

    float baz_1 = foo;
    foo = 1.0;
    GlobalConst phony = msl_padding_global_const;
    idx = int(1);
    t = ConstructBaz(float3x2((1.0).xx, (2.0).xx, (3.0).xx));
    int _e14 = idx;
    idx = asint(asuint(_e14) - asuint(int(1)));
    int _e17 = idx;
    idx = asint(asuint(_e17) + asuint(int(1)));
    SetMatmOnBaz(t, float3x2((6.0).xx, (5.0).xx, (4.0).xx));
    t.m_0 = (9.0).xx;
    int _e33 = idx;
    SetMatVecmOnBaz(t, (90.0).xx, _e33);
    t.m_0[1] = 10.0;
    int _e42 = idx;
    t.m_0[_e42] = 20.0;
    int _e45 = idx;
    SetMatScalarmOnBaz(t, 30.0, _e45, 1);
    int _e49 = idx;
    int _e51 = idx;
    SetMatScalarmOnBaz(t, 40.0, _e49, _e51);
    idx_1 = int(1);
    t_1 = ConstructMatCx2InArray(ZeroValuearray2_float4x2_());
    int _e57 = idx_1;
    idx_1 = asint(asuint(_e57) - asuint(int(1)));
    int _e60 = idx_1;
    idx_1 = asint(asuint(_e60) + asuint(int(1)));
    t_1.am = (__mat4x2_f32[2])ZeroValuearray2_float4x2_();
    t_1.am[0] = (__mat4x2_f32)float4x2((8.0).xx, (7.0).xx, (6.0).xx, (5.0).xx);
    t_1.am[0]._0 = (9.0).xx;
    int _e82 = idx_1;
    __set_col_of_mat4x2_f32(t_1.am[0], _e82, (90.0).xx);
    t_1.am[0]._0.y = 10.0;
    int _e93 = idx_1;
    t_1.am[0]._0[min(uint(_e93), 1u)] = 20.0;
    int _e97 = idx_1;
    __set_el_of_mat4x2_f32(t_1.am[0], _e97, 1, 30.0);
    int _e102 = idx_1;
    int _e104 = idx_1;
    __set_el_of_mat4x2_f32(t_1.am[0], _e102, _e104, 40.0);
    float4x3 _matrix = float4x3(asfloat(bar.Load3(0+0)), asfloat(bar.Load3(0+16)), asfloat(bar.Load3(0+32)), asfloat(bar.Load3(0+48)));
    uint2 arr_1[2] = Constructarray2_uint2_(asuint(bar.Load2(144+0)), asuint(bar.Load2(144+8)));
    float b = asfloat(bar.Load(0+3u*16+0));
    int a_2 = asint(bar.Load(0+(((NagaBufferLengthRW(bar) - 160) / 8) - 2u)*8+160));
    int2 c = asint(qux.Load2(0));
    float _e134 = foo;
    local = _e134;
    float foo_value = local;
    c2_ = Constructarray5_int_(a_2, naga_f2i32(b), int(3), int(4), int(5));
    c2_[min(uint((vi + 1u)), 4u)] = int(42);
    int value_1 = c2_[min(uint(vi), 4u)];
    a = ZeroValuearray5_array10_float__();
    float _e150[5][10] = a;
    local_1 = _e150[4][9];
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
    uint local_2 = (uint)0;
    uint local_3 = (uint)0;
    bool value = (bool)0;
    bool a_1[1] = (bool[1])0;
    bool local_4 = (bool)0;
    S s = (S)0;
    int local_5 = (int)0;
    int local_6 = (int)0;
    Outer thing = (Outer)0;
    Inner inner = (Inner)0;
    int delishus = (int)0;
    int local_7 = (int)0;

    val = 33u;
    arr = Constructarray2_float4_((6.0).xxxx, (7.0).xxxx);
    val = 42u;
    arr = Constructarray2_float4_((1.0).xxxx, (2.0).xxxx);
    s1_ = ZeroValueAssignToMember();
    a1_ = ZeroValuearray4_uint_();
    local_2 = ZeroValueuint();
    local_3 = ZeroValueuint();
    s1_.x = 10u;
    uint _e22 = s1_.x;
    local_2 = _e22;
    a1_[1] = 10u;
    uint _e27 = a1_[1];
    local_3 = _e27;
    value = true;
    bool _e38 = value;
    a_1 = ZeroValuearray1_bool_();
    a_1 = Constructarray1_bool_(_e38);
    bool _e42 = a_1[0];
    local_4 = _e42;
    s = ConstructS(int(42));
    int _e49 = s.m;
    local_5 = _e49;
    if ((ZeroValueOuter().thing != uint(ZeroValueOuter().om_nom_nom.delicious))) {
    }
    local_6 = ZeroValueOuter().om_nom_nom.delicious;
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
    local_7 = _e77;
    return;
}
