#version 310 es

precision highp float;
precision highp int;

struct Packed {
    vec3 member;
    float member_1;
};
const vec2 VEC = vec2(1.0, 2.0);
const mat2x2 MATRIX = mat2x2(VEC, VEC);
const mat2x2 MATRICES[2] = mat2x2[2](MATRIX, MATRIX);
const Packed PACKED = Packed(vec3(1.0, 1.0, 1.0), 2.0);
const Packed PACKED_ARRAY[2] = Packed[2](PACKED, PACKED);

vec4 color = vec4(0.0);

layout(location = 0) out vec4 _fs2p_location0;

void main_1() {
    color = vec4(MATRICES[1][0].y, PACKED_ARRAY[1].member.z, PACKED_ARRAY[0].member_1, 1.0);
    return;
}

void main() {
    main_1();
    vec4 _e1 = color;
    _fs2p_location0 = _e1;
    return;
}

