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

void testNonImplicitCastVectorCast() {
    uint a = 1;
    ivec4 b = ivec4(a);
}

float global;
void privatePointer(inout float a) {}

void ternary(bool a) {
    uint b = a ? 0 : 1u;
    uint c = a ? 0u : 1;

    uint nested = a ? (a ? (a ? 2u : 3) : 4u) : 5;
}

void testMatrixMultiplication(mat4x3 a, mat4x4 b) {
    mat4x3 c = a * b;
}

layout(std430, binding = 0) buffer a_buf {
    float a[];
};

void testLength() {
    int len = a.length();
}

void testConstantLength(float a[4u]) {
    int len = a.length();
}

struct TestStruct { uvec4 array[2]; };
const TestStruct strct = { { uvec4(0), uvec4(1) } };

void indexConstantNonConstantIndex(int i) {
    const uvec4 a = strct.array[i];
}

void testSwizzleWrites(vec3 a) {
    a.zxy.xy = vec2(3.0, 4.0);
    a.rg *= 5.0;
    a.zy++;
}

out vec4 o_color;
void main() {
    privatePointer(global);
    o_color.rgba = vec4(1.0);
}
