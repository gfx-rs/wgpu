#version 310 es

precision highp float;
precision highp int;

struct AlignedWrapper {
    int value;
};
layout(std430) buffer Bar_block_0Vertex {
    mat4x4 matrix;
    mat2x2 matrix_array[2];
    int atom;
    uvec2 arr[2];
    AlignedWrapper data[];
} _group_0_binding_0_vs;


float read_from_private(inout float foo_2) {
    float _e2 = foo_2;
    return _e2;
}

void main() {
    uint vi = uint(gl_VertexID);
    float foo_1 = 0.0;
    int c[5] = int[5](0, 0, 0, 0, 0);
    float baz = foo_1;
    foo_1 = 1.0;
    mat4x4 matrix = _group_0_binding_0_vs.matrix;
    uvec2 arr[2] = _group_0_binding_0_vs.arr;
    float b = _group_0_binding_0_vs.matrix[3][0];
    int a = _group_0_binding_0_vs.data[(uint(_group_0_binding_0_vs.data.length()) - 2u)].value;
    float _e27 = read_from_private(foo_1);
    _group_0_binding_0_vs.matrix[1][2] = 1.0;
    _group_0_binding_0_vs.matrix = mat4x4(vec4(0.0), vec4(1.0), vec4(2.0), vec4(3.0));
    _group_0_binding_0_vs.arr = uvec2[2](uvec2(0u), uvec2(1u));
    _group_0_binding_0_vs.data[1].value = 1;
    c = int[5](a, int(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    int value = c[vi];
    gl_Position = (matrix * vec4(ivec4(value)));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

