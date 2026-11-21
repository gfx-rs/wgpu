#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct StructWithMat
{
    f16mat4x2 m;
};

struct StructWithArrayOfStructOfMat
{
    StructWithMat a[4];
};

struct mat4x2f16_std140
{
    f16vec2 c_0;
    f16vec2 c_1;
    f16vec2 c_2;
    f16vec2 c_3;
};

struct StructWithMat_std140
{
    f16vec2 m_0;
    f16vec2 m_1;
    f16vec2 m_2;
    f16vec2 m_3;
};

struct StructWithArrayOfStructOfMat_std140
{
    StructWithMat_std140 a[4];
};

layout(set = 0, binding = 0, std430) buffer s_m
{
    f16mat4x2 _m0;
} s_m_1;

layout(set = 0, binding = 1, std140) uniform u_m
{
    mat4x2f16_std140 _m0;
} u_m_1;

layout(set = 1, binding = 0, std430) buffer s_sm
{
    StructWithMat _m0;
} s_sm_1;

layout(set = 1, binding = 1, std140) uniform u_sm
{
    StructWithMat_std140 _m0;
} u_sm_1;

layout(set = 2, binding = 0, std430) buffer s_am
{
    f16mat4x2 _m0[4];
} s_am_1;

layout(set = 2, binding = 1, std140) uniform u_am
{
    mat4x2f16_std140 _m0[4];
} u_am_1;

layout(set = 3, binding = 0, std430) buffer s_sasm
{
    StructWithArrayOfStructOfMat _m0;
} s_sasm_1;

layout(set = 3, binding = 1, std140) uniform u_sasm
{
    StructWithArrayOfStructOfMat_std140 _m0;
} u_sasm_1;

void access_m()
{
    int idx = 1;
    idx--;
    f16mat4x2 _103 = f16mat4x2(u_m_1._m0.c_0, u_m_1._m0.c_1, u_m_1._m0.c_2, u_m_1._m0.c_3);
    f16mat4x2 _123 = f16mat4x2(u_m_1._m0.c_0, u_m_1._m0.c_1, u_m_1._m0.c_2, u_m_1._m0.c_3);
    f16mat4x2 _138 = f16mat4x2(u_m_1._m0.c_0, u_m_1._m0.c_1, u_m_1._m0.c_2, u_m_1._m0.c_3);
    s_m_1._m0 = f16mat4x2(u_m_1._m0.c_0, u_m_1._m0.c_1, u_m_1._m0.c_2, u_m_1._m0.c_3);
    s_m_1._m0[0u] = u_m_1._m0.c_0;
    s_m_1._m0[idx] = _103[idx];
    s_m_1._m0[0u].x = u_m_1._m0.c_0.x;
    s_m_1._m0[0u][idx] = u_m_1._m0.c_0[idx];
    s_m_1._m0[idx].x = _123[idx].x;
    s_m_1._m0[idx][idx] = _138[idx][idx];
}

void access_sm()
{
    int idx = 1;
    idx--;
    f16mat4x2 _210 = f16mat4x2(u_sm_1._m0.m_0, u_sm_1._m0.m_1, u_sm_1._m0.m_2, u_sm_1._m0.m_3);
    f16mat4x2 _228 = f16mat4x2(u_sm_1._m0.m_0, u_sm_1._m0.m_1, u_sm_1._m0.m_2, u_sm_1._m0.m_3);
    f16mat4x2 _242 = f16mat4x2(u_sm_1._m0.m_0, u_sm_1._m0.m_1, u_sm_1._m0.m_2, u_sm_1._m0.m_3);
    s_sm_1._m0 = StructWithMat(f16mat4x2(u_sm_1._m0.m_0, u_sm_1._m0.m_1, u_sm_1._m0.m_2, u_sm_1._m0.m_3));
    s_sm_1._m0.m = f16mat4x2(u_sm_1._m0.m_0, u_sm_1._m0.m_1, u_sm_1._m0.m_2, u_sm_1._m0.m_3);
    s_sm_1._m0.m[0u] = u_sm_1._m0.m_0;
    s_sm_1._m0.m[idx] = _210[idx];
    s_sm_1._m0.m[0u].x = u_sm_1._m0.m_0.x;
    s_sm_1._m0.m[0u][idx] = u_sm_1._m0.m_0[idx];
    s_sm_1._m0.m[idx].x = _228[idx].x;
    s_sm_1._m0.m[idx][idx] = _242[idx][idx];
}

f16mat4x2[4] convert_std140_0(mat4x2f16_std140 v[4])
{
    return f16mat4x2[](f16mat4x2(v[0].c_0, v[0].c_1, v[0].c_2, v[0].c_3), f16mat4x2(v[1].c_0, v[1].c_1, v[1].c_2, v[1].c_3), f16mat4x2(v[2].c_0, v[2].c_1, v[2].c_2, v[2].c_3), f16mat4x2(v[3].c_0, v[3].c_1, v[3].c_2, v[3].c_3));
}

void access_am()
{
    int idx = 1;
    idx--;
    f16mat4x2 _376 = f16mat4x2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1, u_am_1._m0[0u].c_2, u_am_1._m0[0u].c_3);
    f16mat4x2 _393 = f16mat4x2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1, u_am_1._m0[idx].c_2, u_am_1._m0[idx].c_3);
    f16mat4x2 _411 = f16mat4x2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1, u_am_1._m0[0u].c_2, u_am_1._m0[0u].c_3);
    f16mat4x2 _425 = f16mat4x2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1, u_am_1._m0[0u].c_2, u_am_1._m0[0u].c_3);
    f16mat4x2 _446 = f16mat4x2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1, u_am_1._m0[idx].c_2, u_am_1._m0[idx].c_3);
    f16mat4x2 _461 = f16mat4x2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1, u_am_1._m0[idx].c_2, u_am_1._m0[idx].c_3);
    s_am_1._m0 = convert_std140_0(u_am_1._m0);
    s_am_1._m0[0u] = f16mat4x2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1, u_am_1._m0[0u].c_2, u_am_1._m0[0u].c_3);
    s_am_1._m0[idx] = f16mat4x2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1, u_am_1._m0[idx].c_2, u_am_1._m0[idx].c_3);
    s_am_1._m0[0u][0u] = u_am_1._m0[0u].c_0;
    s_am_1._m0[0u][idx] = _376[idx];
    s_am_1._m0[idx][0u] = u_am_1._m0[idx].c_0;
    s_am_1._m0[idx][idx] = _393[idx];
    s_am_1._m0[0u][0u].x = u_am_1._m0[0u].c_0.x;
    s_am_1._m0[0u][0u][idx] = u_am_1._m0[0u].c_0[idx];
    s_am_1._m0[0u][idx].x = _411[idx].x;
    s_am_1._m0[0u][idx][idx] = _425[idx][idx];
    s_am_1._m0[idx][0u].x = u_am_1._m0[idx].c_0.x;
    s_am_1._m0[idx][0u][idx] = u_am_1._m0[idx].c_0[idx];
    s_am_1._m0[idx][idx].x = _446[idx].x;
    s_am_1._m0[idx][idx][idx] = _461[idx][idx];
}

StructWithMat[4] convert_std140_1(StructWithMat_std140 v[4])
{
    return StructWithMat[](StructWithMat(f16mat4x2(v[0].m_0, v[0].m_1, v[0].m_2, v[0].m_3)), StructWithMat(f16mat4x2(v[1].m_0, v[1].m_1, v[1].m_2, v[1].m_3)), StructWithMat(f16mat4x2(v[2].m_0, v[2].m_1, v[2].m_2, v[2].m_3)), StructWithMat(f16mat4x2(v[3].m_0, v[3].m_1, v[3].m_2, v[3].m_3)));
}

void access_sasm()
{
    int idx = 1;
    idx--;
    f16mat4x2 _627 = f16mat4x2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1, u_sasm_1._m0.a[0u].m_2, u_sasm_1._m0.a[0u].m_3);
    f16mat4x2 _644 = f16mat4x2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1, u_sasm_1._m0.a[idx].m_2, u_sasm_1._m0.a[idx].m_3);
    f16mat4x2 _662 = f16mat4x2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1, u_sasm_1._m0.a[0u].m_2, u_sasm_1._m0.a[0u].m_3);
    f16mat4x2 _676 = f16mat4x2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1, u_sasm_1._m0.a[0u].m_2, u_sasm_1._m0.a[0u].m_3);
    f16mat4x2 _697 = f16mat4x2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1, u_sasm_1._m0.a[idx].m_2, u_sasm_1._m0.a[idx].m_3);
    f16mat4x2 _712 = f16mat4x2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1, u_sasm_1._m0.a[idx].m_2, u_sasm_1._m0.a[idx].m_3);
    s_sasm_1._m0 = StructWithArrayOfStructOfMat(convert_std140_1(u_sasm_1._m0.a));
    s_sasm_1._m0.a = convert_std140_1(u_sasm_1._m0.a);
    s_sasm_1._m0.a[0u].m = f16mat4x2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1, u_sasm_1._m0.a[0u].m_2, u_sasm_1._m0.a[0u].m_3);
    s_sasm_1._m0.a[idx].m = f16mat4x2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1, u_sasm_1._m0.a[idx].m_2, u_sasm_1._m0.a[idx].m_3);
    s_sasm_1._m0.a[0u].m[0u] = u_sasm_1._m0.a[0u].m_0;
    s_sasm_1._m0.a[0u].m[idx] = _627[idx];
    s_sasm_1._m0.a[idx].m[0u] = u_sasm_1._m0.a[idx].m_0;
    s_sasm_1._m0.a[idx].m[idx] = _644[idx];
    s_sasm_1._m0.a[0u].m[0u].x = u_sasm_1._m0.a[0u].m_0.x;
    s_sasm_1._m0.a[0u].m[0u][idx] = u_sasm_1._m0.a[0u].m_0[idx];
    s_sasm_1._m0.a[0u].m[idx].x = _662[idx].x;
    s_sasm_1._m0.a[0u].m[idx][idx] = _676[idx][idx];
    s_sasm_1._m0.a[idx].m[0u].x = u_sasm_1._m0.a[idx].m_0.x;
    s_sasm_1._m0.a[idx].m[0u][idx] = u_sasm_1._m0.a[idx].m_0[idx];
    s_sasm_1._m0.a[idx].m[idx].x = _697[idx].x;
    s_sasm_1._m0.a[idx].m[idx][idx] = _712[idx][idx];
}

void main()
{
    access_m();
    access_sm();
    access_am();
    access_sasm();
}

