typedef struct { float2 _0; float2 _1; } __mat2x2;
float2 __get_col_of_mat2x2(__mat2x2 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat2x2(__mat2x2 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    }
}
void __set_el_of_mat2x2(__mat2x2 mat, uint idx, uint vec_idx, float value) {
    switch(idx) {
    case 0: { mat._0[vec_idx] = value; break; }
    case 1: { mat._1[vec_idx] = value; break; }
    }
}

typedef struct { float2 _0; float2 _1; float2 _2; float2 _3; } __mat4x2;
float2 __get_col_of_mat4x2(__mat4x2 mat, uint idx) {
    switch(idx) {
    case 0: { return mat._0; }
    case 1: { return mat._1; }
    case 2: { return mat._2; }
    case 3: { return mat._3; }
    default: { return (float2)0; }
    }
}
void __set_col_of_mat4x2(__mat4x2 mat, uint idx, float2 value) {
    switch(idx) {
    case 0: { mat._0 = value; break; }
    case 1: { mat._1 = value; break; }
    case 2: { mat._2 = value; break; }
    case 3: { mat._3 = value; break; }
    }
}
void __set_el_of_mat4x2(__mat4x2 mat, uint idx, uint vec_idx, float value) {
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
    __mat4x2 am[2];
};

struct AssignToMember {
    uint x;
};

GlobalConst ConstructGlobalConst(uint arg0, uint3 arg1, int arg2) {
    GlobalConst ret = (GlobalConst)0;
    ret.a = arg0;
    ret.b = arg1;
    ret.c = arg2;
    return ret;
}

static GlobalConst global_const = ConstructGlobalConst(0u, uint3(0u, 0u, 0u), 0);
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

void test_matrix_within_struct_accesses()
{
    int idx = 1;
    Baz t = ConstructBaz(float3x2((1.0).xx, (2.0).xx, (3.0).xx));

    int _e3 = idx;
    idx = (_e3 - 1);
    float3x2 l0_ = GetMatmOnBaz(baz);
    float2 l1_ = GetMatmOnBaz(baz)[0];
    int _e14 = idx;
    float2 l2_ = GetMatmOnBaz(baz)[_e14];
    float l3_ = GetMatmOnBaz(baz)[0].y;
    int _e25 = idx;
    float l4_ = GetMatmOnBaz(baz)[0][_e25];
    int _e30 = idx;
    float l5_ = GetMatmOnBaz(baz)[_e30].y;
    int _e36 = idx;
    int _e38 = idx;
    float l6_ = GetMatmOnBaz(baz)[_e36][_e38];
    int _e51 = idx;
    idx = (_e51 + 1);
    SetMatmOnBaz(t, float3x2((6.0).xx, (5.0).xx, (4.0).xx));
    t.m_0 = (9.0).xx;
    int _e66 = idx;
    SetMatVecmOnBaz(t, (90.0).xx, _e66);
    t.m_0[1] = 10.0;
    int _e76 = idx;
    t.m_0[_e76] = 20.0;
    int _e80 = idx;
    SetMatScalarmOnBaz(t, 30.0, _e80, 1);
    int _e85 = idx;
    int _e87 = idx;
    SetMatScalarmOnBaz(t, 40.0, _e85, _e87);
    return;
}

MatCx2InArray ConstructMatCx2InArray(float4x2 arg0[2]) {
    MatCx2InArray ret = (MatCx2InArray)0;
    ret.am = (__mat4x2[2])arg0;
    return ret;
}

typedef float4x2 ret_ZeroValuearray2_float4x2_[2];
ret_ZeroValuearray2_float4x2_ ZeroValuearray2_float4x2_() {
    return (float4x2[2])0;
}

void test_matrix_within_array_within_struct_accesses()
{
    int idx_1 = 1;
    MatCx2InArray t_1 = ConstructMatCx2InArray(ZeroValuearray2_float4x2_());

    int _e3 = idx_1;
    idx_1 = (_e3 - 1);
    float4x2 l0_1[2] = ((float4x2[2])nested_mat_cx2_.am);
    float4x2 l1_1 = ((float4x2)nested_mat_cx2_.am[0]);
    float2 l2_1 = nested_mat_cx2_.am[0]._0;
    int _e20 = idx_1;
    float2 l3_1 = __get_col_of_mat4x2(nested_mat_cx2_.am[0], _e20);
    float l4_1 = nested_mat_cx2_.am[0]._0.y;
    int _e33 = idx_1;
    float l5_1 = nested_mat_cx2_.am[0]._0[_e33];
    int _e39 = idx_1;
    float l6_1 = __get_col_of_mat4x2(nested_mat_cx2_.am[0], _e39).y;
    int _e46 = idx_1;
    int _e48 = idx_1;
    float l7_ = __get_col_of_mat4x2(nested_mat_cx2_.am[0], _e46)[_e48];
    int _e55 = idx_1;
    idx_1 = (_e55 + 1);
    t_1.am = (__mat4x2[2])ZeroValuearray2_float4x2_();
    t_1.am[0] = (__mat4x2)float4x2((8.0).xx, (7.0).xx, (6.0).xx, (5.0).xx);
    t_1.am[0]._0 = (9.0).xx;
    int _e77 = idx_1;
    __set_col_of_mat4x2(t_1.am[0], _e77, (90.0).xx);
    t_1.am[0]._0.y = 10.0;
    int _e89 = idx_1;
    t_1.am[0]._0[_e89] = 20.0;
    int _e94 = idx_1;
    __set_el_of_mat4x2(t_1.am[0], _e94, 1, 30.0);
    int _e100 = idx_1;
    int _e102 = idx_1;
    __set_el_of_mat4x2(t_1.am[0], _e100, _e102, 40.0);
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

typedef int ret_Constructarray5_int_[5];
ret_Constructarray5_int_ Constructarray5_int_(int arg0, int arg1, int arg2, int arg3, int arg4) {
    int ret[5] = { arg0, arg1, arg2, arg3, arg4 };
    return ret;
}

typedef float ret_ZeroValuearray5_array10_float__[5][10];
ret_ZeroValuearray5_array10_float__ ZeroValuearray5_array10_float__() {
    return (float[5][10])0;
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
    test_matrix_within_struct_accesses();
    test_matrix_within_array_within_struct_accesses();
    float4x3 _matrix = float4x3(asfloat(bar.Load3(0+0)), asfloat(bar.Load3(0+16)), asfloat(bar.Load3(0+32)), asfloat(bar.Load3(0+48)));
    uint2 arr_1[2] = Constructarray2_uint2_(asuint(bar.Load2(144+0)), asuint(bar.Load2(144+8)));
    float b = asfloat(bar.Load(0+3u*16+0));
    int a_1 = asint(bar.Load(0+(((NagaBufferLengthRW(bar) - 160) / 8) - 2u)*8+160));
    int2 c = asint(qux.Load2(0));
    const float _e33 = read_from_private(foo);
    c2_ = Constructarray5_int_(a_1, int(b), 3, 4, 5);
    c2_[(vi + 1u)] = 42;
    int value = c2_[vi];
    const float _e47 = test_arr_as_arg(ZeroValuearray5_array10_float__());
    return float4(mul(float4((value).xxxx), _matrix), 2.0);
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
    bar.Store(0+8+160, asuint(1));
    qux.Store2(0, asuint(ZeroValueint2()));
    return (0.0).xxxx;
}

[numthreads(1, 1, 1)]
void assign_through_ptr()
{
    uint val = 33u;
    float4 arr[2] = Constructarray2_float4_((6.0).xxxx, (7.0).xxxx);

    assign_through_ptr_fn(val);
    assign_array_through_ptr_fn(arr);
    return;
}

[numthreads(1, 1, 1)]
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
