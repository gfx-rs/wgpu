#version 450

layout(set = 0, binding = 0) buffer testBufferBlock {
    uint[] data;
} testBuffer;

layout(set = 0, binding = 2) readonly buffer testBufferReadOnlyBlock {
    uint[] data;
} testBufferReadOnly;

void main() {
    uint a = testBuffer.data[0];
    testBuffer.data[1] = 2;

    uint b = testBufferReadOnly.data[0];
}
