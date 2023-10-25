#version 450

const int NUM_VECS = 42;
layout(std140, set = 1, binding = 0) uniform Data {
    vec4 vecs[NUM_VECS];
};

vec4 function() {
    vec4 sum = vec4(0);
    for (int i = 0; i < NUM_VECS; i++) {
        sum += vecs[i];
    }
    return sum;
}

void main() {}
