#version 460
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _21_20
{
    float _m0[];
} _20;

coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseA> _14 = coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseA>(0.0);
coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseB> _17 = coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseB>(0.0);

void main()
{
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _28 = coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _31 = coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator>(0.0);
    coopmat<float, gl_ScopeSubgroup, 8u, 8u, gl_MatrixUseAccumulator> _37;
    coopMatLoad(_37, _20._m0, 4u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    _28 = _37;
    _31 = coopMatMulAdd(_14, _17, _28, 0);
    coopMatStore(_31, _20._m0, 0u, 8u, gl_CooperativeMatrixLayoutColumnMajor);
    _28 = _31;
}

