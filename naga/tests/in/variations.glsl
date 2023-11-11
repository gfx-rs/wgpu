#version 460 core

layout(set = 0, binding = 0) uniform textureCube texCube;
layout(set = 0, binding = 1) uniform sampler samp;

void main() {
    ivec2 sizeCube = textureSize(samplerCube(texCube, samp), 0);
    float a = ceil(1.0);
}
