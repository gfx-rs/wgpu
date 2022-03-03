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

void testBinOpMatMat(mat3 a, mat3 b) {
    mat3 v;
    bool c;
    v = a / b;
    v = a * b;
    v = a + b;
    v = a - b;
    c = a == b;
    c = a != b;
}

void testBinOpMatFloat(float a, mat3 b) {
    mat3 v;
    v = a / b;
    v = a * b;
    v = a + b;
    v = a - b;

    v = b / a;
    v = b * a;
    v = b + a;
    v = b - a;
}

void testUnaryOpMat(mat3 a) {
    mat3 v;
    v = -a;
    v = --a;
    v = a--;
}

void testStructConstructor() {
    struct BST {
        int data;
    };

    BST tree = BST(1);
}

void testNonScalarToScalarConstructor() {
    float f = float(mat2(1.0));
}

void testArrayConstructor() {
    float tree[1] = float[1](0.0);
}

void testFreestandingConstructor() {
    vec4(1.0);
}

float global;
void privatePointer(inout float a) {}

out vec4 o_color;
void main() {
    privatePointer(global);
    o_color.rgba = vec4(1.0);
}
