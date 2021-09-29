#version 310 es

precision highp float;
precision highp int;

layout(std430) buffer Bar_block_0Vs {
    mat4x4 matrix;
    int atom;
    uvec2 arr[2];
    int data[];
} _group_0_binding_0;


float read_from_private(inout float foo2) {
    float _e2 = foo2;
    return _e2;
}

void main() {
    uint vi = uint(gl_VertexID);
    float foo1 = 0.0;
    int c[5];
    float baz = foo1;
    foo1 = 1.0;
    mat4x4 matrix = _group_0_binding_0.matrix;
    uvec2 arr[2] = _group_0_binding_0.arr;
    float b = _group_0_binding_0.matrix[3][0];
    int a = _group_0_binding_0.data[(uint(_group_0_binding_0.data.length()) - 2u)];
    float _e25 = read_from_private(foo1);
    _group_0_binding_0.matrix[1][2] = 1.0;
    _group_0_binding_0.matrix = mat4x4(vec4(0.0), vec4(1.0), vec4(2.0), vec4(3.0));
    _group_0_binding_0.arr = uvec2[2](uvec2(0u), uvec2(1u));
    c = int[5](a, int(b), 3, 4, 5);
    c[(vi + 1u)] = 42;
    int value = c[vi];
    gl_Position = (matrix * vec4(ivec4(value)));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}

