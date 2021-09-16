#version 440 core

void testBinOpVecFloat(vec4 a, float b) {
    vec4 v;
    v = a * 2.0;
    v = a / 2.0;
    v = a + 2.0;
    v = a - 2.0;
}

void testBinOpFloatVec(vec4 a, float b) {
    vec4 v;
    v = a * b;
    v = a / b;
    v = a + b;
    v = a - b;
}

void testBinOpIVecInt(ivec4 a, int b) {
    ivec4 v;
    v = a * b;
    v = a / b;
    v = a + b;
    v = a - b;
    v = a & b;
    v = a | b;
    v = a ^ b;
    v = a >> b;
    v = a << b;
}

void testBinOpIntIVec(int a, ivec4 b) {
    ivec4 v;
    v = a * b;
    v = a + b;
    v = a - b;
    v = a & b;
    v = a | b;
    v = a ^ b;
}

void testBinOpUVecUint(uvec4 a, uint b) {
    uvec4 v;
    v = a * b;
    v = a / b;
    v = a + b;
    v = a - b;
    v = a & b;
    v = a | b;
    v = a ^ b;
    v = a >> b;
    v = a << b;
}

void testBinOpUintUVec(uint a, uvec4 b) {
    uvec4 v;
    v = a * b;
    v = a + b;
    v = a - b;
    v = a & b;
    v = a | b;
    v = a ^ b;
}

out vec4 o_color;
void main() {
    o_color.rgba = vec4(1.0);
}
