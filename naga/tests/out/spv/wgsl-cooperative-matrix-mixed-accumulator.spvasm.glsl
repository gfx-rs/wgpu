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
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _13_12
{
    float16_t _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float _m0[];
} _15;

void main()
{
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _26 = coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator>(0.0);
    coopmat<float16_t, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseA> _34;
    coopMatLoad(_34, _12._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseB> _37;
    coopMatLoad(_37, _12._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _40;
    coopMatLoad(_40, _15._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    _26 = _40;
    _26 = coopMatMulAdd(_34, _37, _26, 0);
    coopMatStore(_26, _15._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
}

