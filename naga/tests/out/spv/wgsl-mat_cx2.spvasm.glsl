#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct StructWithMat
{
    mat2 m;
};

struct StructWithArrayOfStructOfMat
{
    StructWithMat a[4];
};

struct mat2x2f32_std140
{
    vec2 c_0;
    vec2 c_1;
};

struct StructWithMat_std140
{
    vec2 m_0;
    vec2 m_1;
};

struct StructWithArrayOfStructOfMat_std140
{
    StructWithMat_std140 a[4];
};

layout(set = 0, binding = 0, std430) buffer s_m
{
    mat2 _m0;
} s_m_1;

layout(set = 0, binding = 1, std140) uniform u_m
{
    mat2x2f32_std140 _m0;
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
    mat2 _m0[4];
} s_am_1;

layout(set = 2, binding = 1, std140) uniform u_am
{
    mat2x2f32_std140 _m0[4];
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
    mat2 _93 = mat2(u_m_1._m0.c_0, u_m_1._m0.c_1);
    mat2 _109 = mat2(u_m_1._m0.c_0, u_m_1._m0.c_1);
    mat2 _120 = mat2(u_m_1._m0.c_0, u_m_1._m0.c_1);
    s_m_1._m0 = mat2(u_m_1._m0.c_0, u_m_1._m0.c_1);
    s_m_1._m0[0u] = u_m_1._m0.c_0;
    s_m_1._m0[idx] = _93[idx];
    s_m_1._m0[0u].x = u_m_1._m0.c_0.x;
    s_m_1._m0[0u][idx] = u_m_1._m0.c_0[idx];
    s_m_1._m0[idx].x = _109[idx].x;
    s_m_1._m0[idx][idx] = _120[idx][idx];
}

void access_sm()
{
    int idx = 1;
    idx--;
    mat2 _182 = mat2(u_sm_1._m0.m_0, u_sm_1._m0.m_1);
    mat2 _196 = mat2(u_sm_1._m0.m_0, u_sm_1._m0.m_1);
    mat2 _206 = mat2(u_sm_1._m0.m_0, u_sm_1._m0.m_1);
    s_sm_1._m0 = StructWithMat(mat2(u_sm_1._m0.m_0, u_sm_1._m0.m_1));
    s_sm_1._m0.m = mat2(u_sm_1._m0.m_0, u_sm_1._m0.m_1);
    s_sm_1._m0.m[0u] = u_sm_1._m0.m_0;
    s_sm_1._m0.m[idx] = _182[idx];
    s_sm_1._m0.m[0u].x = u_sm_1._m0.m_0.x;
    s_sm_1._m0.m[0u][idx] = u_sm_1._m0.m_0[idx];
    s_sm_1._m0.m[idx].x = _196[idx].x;
    s_sm_1._m0.m[idx][idx] = _206[idx][idx];
}

mat2[4] convert_std140_0(mat2x2f32_std140 v[4])
{
    return mat2[](mat2(v[0].c_0, v[0].c_1), mat2(v[1].c_0, v[1].c_1), mat2(v[2].c_0, v[2].c_1), mat2(v[3].c_0, v[3].c_1));
}

void access_am()
{
    int idx = 1;
    idx--;
    mat2 _320 = mat2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1);
    mat2 _333 = mat2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1);
    mat2 _347 = mat2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1);
    mat2 _357 = mat2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1);
    mat2 _374 = mat2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1);
    mat2 _385 = mat2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1);
    s_am_1._m0 = convert_std140_0(u_am_1._m0);
    s_am_1._m0[0u] = mat2(u_am_1._m0[0u].c_0, u_am_1._m0[0u].c_1);
    s_am_1._m0[idx] = mat2(u_am_1._m0[idx].c_0, u_am_1._m0[idx].c_1);
    s_am_1._m0[0u][0u] = u_am_1._m0[0u].c_0;
    s_am_1._m0[0u][idx] = _320[idx];
    s_am_1._m0[idx][0u] = u_am_1._m0[idx].c_0;
    s_am_1._m0[idx][idx] = _333[idx];
    s_am_1._m0[0u][0u].x = u_am_1._m0[0u].c_0.x;
    s_am_1._m0[0u][0u][idx] = u_am_1._m0[0u].c_0[idx];
    s_am_1._m0[0u][idx].x = _347[idx].x;
    s_am_1._m0[0u][idx][idx] = _357[idx][idx];
    s_am_1._m0[idx][0u].x = u_am_1._m0[idx].c_0.x;
    s_am_1._m0[idx][0u][idx] = u_am_1._m0[idx].c_0[idx];
    s_am_1._m0[idx][idx].x = _374[idx].x;
    s_am_1._m0[idx][idx][idx] = _385[idx][idx];
}

StructWithMat[4] convert_std140_1(StructWithMat_std140 v[4])
{
    return StructWithMat[](StructWithMat(mat2(v[0].m_0, v[0].m_1)), StructWithMat(mat2(v[1].m_0, v[1].m_1)), StructWithMat(mat2(v[2].m_0, v[2].m_1)), StructWithMat(mat2(v[3].m_0, v[3].m_1)));
}

void access_sasm()
{
    int idx = 1;
    idx--;
    mat2 _531 = mat2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1);
    mat2 _544 = mat2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1);
    mat2 _558 = mat2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1);
    mat2 _568 = mat2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1);
    mat2 _585 = mat2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1);
    mat2 _596 = mat2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1);
    s_sasm_1._m0 = StructWithArrayOfStructOfMat(convert_std140_1(u_sasm_1._m0.a));
    s_sasm_1._m0.a = convert_std140_1(u_sasm_1._m0.a);
    s_sasm_1._m0.a[0u].m = mat2(u_sasm_1._m0.a[0u].m_0, u_sasm_1._m0.a[0u].m_1);
    s_sasm_1._m0.a[idx].m = mat2(u_sasm_1._m0.a[idx].m_0, u_sasm_1._m0.a[idx].m_1);
    s_sasm_1._m0.a[0u].m[0u] = u_sasm_1._m0.a[0u].m_0;
    s_sasm_1._m0.a[0u].m[idx] = _531[idx];
    s_sasm_1._m0.a[idx].m[0u] = u_sasm_1._m0.a[idx].m_0;
    s_sasm_1._m0.a[idx].m[idx] = _544[idx];
    s_sasm_1._m0.a[0u].m[0u].x = u_sasm_1._m0.a[0u].m_0.x;
    s_sasm_1._m0.a[0u].m[0u][idx] = u_sasm_1._m0.a[0u].m_0[idx];
    s_sasm_1._m0.a[0u].m[idx].x = _558[idx].x;
    s_sasm_1._m0.a[0u].m[idx][idx] = _568[idx][idx];
    s_sasm_1._m0.a[idx].m[0u].x = u_sasm_1._m0.a[idx].m_0.x;
    s_sasm_1._m0.a[idx].m[0u][idx] = u_sasm_1._m0.a[idx].m_0[idx];
    s_sasm_1._m0.a[idx].m[idx].x = _585[idx].x;
    s_sasm_1._m0.a[idx].m[idx][idx] = _596[idx][idx];
}

void main()
{
    access_m();
    access_sm();
    access_am();
    access_sasm();
}

