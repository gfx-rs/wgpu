#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _14_13
{
    float16_t _m0[];
} _13;

layout(set = 0, binding = 1, std430) buffer _17_16
{
    float _m0[];
} _16;

void main()
{
    coopmat<float, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseAccumulator> _27 = coopmat<float, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseAccumulator>(0.0);
    coopmat<float16_t, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseA> _35;
    coopMatLoad(_35, _13._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseB> _38;
    coopMatLoad(_38, _13._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseAccumulator> _41;
    coopMatLoad(_41, _16._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    _27 = _41;
    _27 = coopMatMulAdd(_35, _38, _27, 0);
    coopMatStore(_27, _16._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
}

