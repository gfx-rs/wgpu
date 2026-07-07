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

struct std140_mat4x2_f16_
{
    f16vec2 col0;
    f16vec2 col1;
    f16vec2 col2;
    f16vec2 col3;
};

struct std140_StructWithMat
{
    f16vec2 m_col0;
    f16vec2 m_col1;
    f16vec2 m_col2;
    f16vec2 m_col3;
};

struct std140_StructWithArrayOfStructOfMat
{
    std140_StructWithMat a[4];
};

layout(set = 0, binding = 0, std430) buffer s_m
{
    f16mat4x2 _m0;
} s_m_1;

layout(set = 0, binding = 1, std140) uniform u_m
{
    std140_mat4x2_f16_ _m0;
} u_m_1;

layout(set = 1, binding = 0, std430) buffer s_sm
{
    StructWithMat _m0;
} s_sm_1;

layout(set = 1, binding = 1, std140) uniform u_sm
{
    std140_StructWithMat _m0;
} u_sm_1;

layout(set = 2, binding = 0, std430) buffer s_am
{
    f16mat4x2 _m0[4];
} s_am_1;

layout(set = 2, binding = 1, std140) uniform u_am
{
    std140_mat4x2_f16_ _m0[4];
} u_am_1;

layout(set = 3, binding = 0, std430) buffer s_sasm
{
    StructWithArrayOfStructOfMat _m0;
} s_sasm_1;

layout(set = 3, binding = 1, std140) uniform u_sasm
{
    std140_StructWithArrayOfStructOfMat _m0;
} u_sasm_1;

f16mat4x2 mat4x2_f16_from_std140(std140_mat4x2_f16_ _45)
{
    return f16mat4x2(_45.col0, _45.col1, _45.col2, _45.col3);
}

f16vec2 mat4x2_f16_get_column(f16mat4x2 _53, uint _54)
{
    f16vec2 _67;
    switch (_54)
    {
        case 0u:
        {
            _67 = _53[0];
            break;
        }
        case 1u:
        {
            _67 = _53[1];
            break;
        }
        case 2u:
        {
            _67 = _53[2];
            break;
        }
        case 3u:
        {
            _67 = _53[3];
            break;
        }
        default:
        {
            break; // unreachable workaround
        }
    }
    return _67;
}

void access_m()
{
    int idx = 1;
    idx--;
    s_m_1._m0 = mat4x2_f16_from_std140(u_m_1._m0);
    s_m_1._m0[0u] = u_m_1._m0.col0;
    s_m_1._m0[idx] = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_m_1._m0), uint(idx));
    s_m_1._m0[0u].x = u_m_1._m0.col0.x;
    s_m_1._m0[0u][idx] = u_m_1._m0.col0[idx];
    s_m_1._m0[idx].x = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_m_1._m0), uint(idx)).x;
    s_m_1._m0[idx][idx] = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_m_1._m0), uint(idx))[idx];
}

StructWithMat StructWithMat_from_std140(std140_StructWithMat _145)
{
    return StructWithMat(f16mat4x2(_145.m_col0, _145.m_col1, _145.m_col2, _145.m_col3));
}

void access_sm()
{
    int idx = 1;
    idx--;
    s_sm_1._m0 = StructWithMat_from_std140(u_sm_1._m0);
    s_sm_1._m0.m = f16mat4x2(u_sm_1._m0.m_col0, u_sm_1._m0.m_col1, u_sm_1._m0.m_col2, u_sm_1._m0.m_col3);
    s_sm_1._m0.m[0u] = u_sm_1._m0.m_col0;
    s_sm_1._m0.m[idx] = mat4x2_f16_get_column(f16mat4x2(u_sm_1._m0.m_col0, u_sm_1._m0.m_col1, u_sm_1._m0.m_col2, u_sm_1._m0.m_col3), uint(idx));
    s_sm_1._m0.m[0u].x = u_sm_1._m0.m_col0.x;
    s_sm_1._m0.m[0u][idx] = u_sm_1._m0.m_col0[idx];
    s_sm_1._m0.m[idx].x = mat4x2_f16_get_column(f16mat4x2(u_sm_1._m0.m_col0, u_sm_1._m0.m_col1, u_sm_1._m0.m_col2, u_sm_1._m0.m_col3), uint(idx)).x;
    s_sm_1._m0.m[idx][idx] = mat4x2_f16_get_column(f16mat4x2(u_sm_1._m0.m_col0, u_sm_1._m0.m_col1, u_sm_1._m0.m_col2, u_sm_1._m0.m_col3), uint(idx))[idx];
}

f16mat4x2[4] array_mat4x2_f16_4_from_std140(std140_mat4x2_f16_ _258[4])
{
    return f16mat4x2[](mat4x2_f16_from_std140(_258[0]), mat4x2_f16_from_std140(_258[1]), mat4x2_f16_from_std140(_258[2]), mat4x2_f16_from_std140(_258[3]));
}

void access_am()
{
    int idx = 1;
    idx--;
    s_am_1._m0 = array_mat4x2_f16_4_from_std140(u_am_1._m0);
    s_am_1._m0[0u] = mat4x2_f16_from_std140(u_am_1._m0[0u]);
    s_am_1._m0[idx] = mat4x2_f16_from_std140(u_am_1._m0[idx]);
    s_am_1._m0[0u][0u] = u_am_1._m0[0u].col0;
    s_am_1._m0[0u][idx] = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_am_1._m0[0u]), uint(idx));
    s_am_1._m0[idx][0u] = u_am_1._m0[idx].col0;
    s_am_1._m0[idx][idx] = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_am_1._m0[idx]), uint(idx));
    s_am_1._m0[0u][0u].x = u_am_1._m0[0u].col0.x;
    s_am_1._m0[0u][0u][idx] = u_am_1._m0[0u].col0[idx];
    s_am_1._m0[0u][idx].x = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_am_1._m0[0u]), uint(idx)).x;
    s_am_1._m0[0u][idx][idx] = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_am_1._m0[0u]), uint(idx))[idx];
    s_am_1._m0[idx][0u].x = u_am_1._m0[idx].col0.x;
    s_am_1._m0[idx][0u][idx] = u_am_1._m0[idx].col0[idx];
    s_am_1._m0[idx][idx].x = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_am_1._m0[idx]), uint(idx)).x;
    s_am_1._m0[idx][idx][idx] = mat4x2_f16_get_column(mat4x2_f16_from_std140(u_am_1._m0[idx]), uint(idx))[idx];
}

StructWithMat[4] array_StructWithMat_4_from_std140(std140_StructWithMat _435[4])
{
    return StructWithMat[](StructWithMat_from_std140(_435[0]), StructWithMat_from_std140(_435[1]), StructWithMat_from_std140(_435[2]), StructWithMat_from_std140(_435[3]));
}

StructWithArrayOfStructOfMat StructWithArrayOfStructOfMat_from_std140(std140_StructWithArrayOfStructOfMat _430)
{
    return StructWithArrayOfStructOfMat(array_StructWithMat_4_from_std140(_430.a));
}

void access_sasm()
{
    int idx = 1;
    idx--;
    s_sasm_1._m0 = StructWithArrayOfStructOfMat_from_std140(u_sasm_1._m0);
    s_sasm_1._m0.a = array_StructWithMat_4_from_std140(u_sasm_1._m0.a);
    s_sasm_1._m0.a[0u].m = f16mat4x2(u_sasm_1._m0.a[0u].m_col0, u_sasm_1._m0.a[0u].m_col1, u_sasm_1._m0.a[0u].m_col2, u_sasm_1._m0.a[0u].m_col3);
    s_sasm_1._m0.a[idx].m = f16mat4x2(u_sasm_1._m0.a[idx].m_col0, u_sasm_1._m0.a[idx].m_col1, u_sasm_1._m0.a[idx].m_col2, u_sasm_1._m0.a[idx].m_col3);
    s_sasm_1._m0.a[0u].m[0u] = u_sasm_1._m0.a[0u].m_col0;
    s_sasm_1._m0.a[0u].m[idx] = mat4x2_f16_get_column(f16mat4x2(u_sasm_1._m0.a[0u].m_col0, u_sasm_1._m0.a[0u].m_col1, u_sasm_1._m0.a[0u].m_col2, u_sasm_1._m0.a[0u].m_col3), uint(idx));
    s_sasm_1._m0.a[idx].m[0u] = u_sasm_1._m0.a[idx].m_col0;
    s_sasm_1._m0.a[idx].m[idx] = mat4x2_f16_get_column(f16mat4x2(u_sasm_1._m0.a[idx].m_col0, u_sasm_1._m0.a[idx].m_col1, u_sasm_1._m0.a[idx].m_col2, u_sasm_1._m0.a[idx].m_col3), uint(idx));
    s_sasm_1._m0.a[0u].m[0u].x = u_sasm_1._m0.a[0u].m_col0.x;
    s_sasm_1._m0.a[0u].m[0u][idx] = u_sasm_1._m0.a[0u].m_col0[idx];
    s_sasm_1._m0.a[0u].m[idx].x = mat4x2_f16_get_column(f16mat4x2(u_sasm_1._m0.a[0u].m_col0, u_sasm_1._m0.a[0u].m_col1, u_sasm_1._m0.a[0u].m_col2, u_sasm_1._m0.a[0u].m_col3), uint(idx)).x;
    s_sasm_1._m0.a[0u].m[idx][idx] = mat4x2_f16_get_column(f16mat4x2(u_sasm_1._m0.a[0u].m_col0, u_sasm_1._m0.a[0u].m_col1, u_sasm_1._m0.a[0u].m_col2, u_sasm_1._m0.a[0u].m_col3), uint(idx))[idx];
    s_sasm_1._m0.a[idx].m[0u].x = u_sasm_1._m0.a[idx].m_col0.x;
    s_sasm_1._m0.a[idx].m[0u][idx] = u_sasm_1._m0.a[idx].m_col0[idx];
    s_sasm_1._m0.a[idx].m[idx].x = mat4x2_f16_get_column(f16mat4x2(u_sasm_1._m0.a[idx].m_col0, u_sasm_1._m0.a[idx].m_col1, u_sasm_1._m0.a[idx].m_col2, u_sasm_1._m0.a[idx].m_col3), uint(idx)).x;
    s_sasm_1._m0.a[idx].m[idx][idx] = mat4x2_f16_get_column(f16mat4x2(u_sasm_1._m0.a[idx].m_col0, u_sasm_1._m0.a[idx].m_col1, u_sasm_1._m0.a[idx].m_col2, u_sasm_1._m0.a[idx].m_col3), uint(idx))[idx];
}

void main()
{
    access_m();
    access_sm();
    access_am();
    access_sasm();
}

