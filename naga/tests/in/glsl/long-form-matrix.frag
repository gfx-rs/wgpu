// ISSUE: #1064
#version 450

void main() {
    // Sane ways to build a matrix
    mat2 splat = mat2(1);
    mat2 normal = mat2(vec2(1), vec2(2));
    mat2x4 from_matrix = mat2x4(mat3(1.0));

    // This is a little bit weirder but still makes some sense
    // Since this matrix has 2 rows we take two numbers to make a column
    // and we do this twice because we 2 columns.
    // Final result in wgsl should be:
    // mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0))
    mat2 a = mat2(1, 2, 3, 4);

    // ???
    // Glsl has decided that for it's matrix constructor arguments it doesn't
    // take them as is but instead flattens them so the `b` matrix is
    // equivalent to the `a` matrix but in value and semantics
    mat2 b = mat2(1, vec2(2, 3), 4);
    mat3 c = mat3(1, 2, 3, vec3(1), vec3(1));
    mat3 d = mat3(vec2(2), 1, vec3(1), vec3(1));
    mat4 e = mat4(vec2(2), vec4(1), vec2(2), vec4(1), vec4(1));
}
