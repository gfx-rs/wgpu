/////////////////////////////////////
// Entry point: "tile_16x8" (comp) //
/////////////////////////////////////
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

layout(set = 0, binding = 0, std430) buffer _15_14
{
    float16_t _m0[];
} _14;

layout(set = 0, binding = 1, std430) buffer _18_17
{
    float _m0[];
} _17;

void main()
{
    coopmat<float, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseAccumulator> _28 = coopmat<float, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseAccumulator>(0.0);
    coopmat<float16_t, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseA> _36;
    coopMatLoad(_36, _14._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseB> _39;
    coopMatLoad(_39, _14._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, 8u, 16u, gl_MatrixUseAccumulator> _42;
    coopMatLoad(_42, _17._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    _28 = _42;
    _28 = coopMatMulAdd(_36, _39, _28, 0);
    coopMatStore(_28, _17._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
}


/////////////////////////////////////
// Entry point: "tile_8x16" (comp) //
/////////////////////////////////////
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

layout(set = 0, binding = 0, std430) buffer _15_14
{
    float16_t _m0[];
} _14;

layout(set = 0, binding = 1, std430) buffer _18_17
{
    float _m0[];
} _17;

void main()
{
    coopmat<float, gl_ScopeSubgroup, 16u, 8u, gl_MatrixUseAccumulator> _51 = coopmat<float, gl_ScopeSubgroup, 16u, 8u, gl_MatrixUseAccumulator>(0.0);
    coopmat<float16_t, gl_ScopeSubgroup, 16u, 16u, gl_MatrixUseA> _57;
    coopMatLoad(_57, _14._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float16_t, gl_ScopeSubgroup, 16u, 8u, gl_MatrixUseB> _60;
    coopMatLoad(_60, _14._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    coopmat<float, gl_ScopeSubgroup, 16u, 8u, gl_MatrixUseAccumulator> _62;
    coopMatLoad(_62, _17._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
    _51 = _62;
    _51 = coopMatMulAdd(_57, _60, _51, 0);
    coopMatStore(_51, _17._m0, 0u, 16u, gl_CooperativeMatrixLayoutColumnMajor);
}

