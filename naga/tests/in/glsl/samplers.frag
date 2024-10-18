#version 440 core
precision mediump float;

layout(set = 1, binding = 0) uniform texture1D tex1D;
layout(set = 1, binding = 1) uniform texture1DArray tex1DArray;
layout(set = 1, binding = 2) uniform texture2D tex2D;
layout(set = 1, binding = 3) uniform texture2DArray tex2DArray;
layout(set = 1, binding = 4) uniform textureCube texCube;
layout(set = 1, binding = 5) uniform textureCubeArray texCubeArray;
layout(set = 1, binding = 6) uniform texture3D tex3D;
layout(set = 1, binding = 7) uniform sampler samp;

// WGSL doesn't have 1D depth samplers.
#define HAS_1D_DEPTH_TEXTURES 0

#if HAS_1D_DEPTH_TEXTURES
layout(set = 1, binding = 10) uniform texture1D tex1DShadow;
layout(set = 1, binding = 11) uniform texture1DArray tex1DArrayShadow;
#endif

layout(set = 1, binding = 12) uniform texture2D tex2DShadow;
layout(set = 1, binding = 13) uniform texture2DArray tex2DArrayShadow;
layout(set = 1, binding = 14) uniform textureCube texCubeShadow;
layout(set = 1, binding = 15) uniform textureCubeArray texCubeArrayShadow;
layout(set = 1, binding = 16) uniform texture3D tex3DShadow;
layout(set = 1, binding = 17) uniform samplerShadow sampShadow;

layout(binding = 18) uniform texture2DMS tex2DMS;
layout(binding = 19) uniform texture2DMSArray tex2DMSArray;

// Conventions for readability:
//   1.0 = Shadow Ref
//   2.0 = LOD Bias
//   3.0 = Explicit LOD
//   4.0 = Grad Derivatives
//   5 = Offset
//   6.0 = Proj W

void testTex1D(in float coord) {
    int size1D = textureSize(sampler1D(tex1D, samp), 0);
    int levels = textureQueryLevels(sampler1D(tex1D, samp));
    vec4 c;
    c = texture(sampler1D(tex1D, samp), coord);
    c = texture(sampler1D(tex1D, samp), coord, 2.0);
    c = textureGrad(sampler1D(tex1D, samp), coord, 4.0, 4.0);
    c = textureGradOffset(sampler1D(tex1D, samp), coord, 4.0, 4.0, 5);
    c = textureLod(sampler1D(tex1D, samp), coord, 3.0);
    c = textureLodOffset(sampler1D(tex1D, samp), coord, 3.0, 5);
    c = textureOffset(sampler1D(tex1D, samp), coord, 5);
    c = textureOffset(sampler1D(tex1D, samp), coord, 5, 2.0);
    c = textureProj(sampler1D(tex1D, samp), vec2(coord, 6.0));
    c = textureProj(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0));
    c = textureProj(sampler1D(tex1D, samp), vec2(coord, 6.0), 2.0);
    c = textureProj(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 2.0);
    c = textureProjGrad(sampler1D(tex1D, samp), vec2(coord, 6.0), 4.0, 4.0);
    c = textureProjGrad(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 4.0, 4.0);
    c = textureProjGradOffset(sampler1D(tex1D, samp), vec2(coord, 6.0), 4.0, 4.0, 5);
    c = textureProjGradOffset(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 4.0, 4.0, 5);
    c = textureProjLod(sampler1D(tex1D, samp), vec2(coord, 6.0), 3.0);
    c = textureProjLod(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 3.0);
    c = textureProjLodOffset(sampler1D(tex1D, samp), vec2(coord, 6.0), 3.0, 5);
    c = textureProjLodOffset(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 3.0, 5);
    c = textureProjOffset(sampler1D(tex1D, samp), vec2(coord, 6.0), 5);
    c = textureProjOffset(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 5);
    c = textureProjOffset(sampler1D(tex1D, samp), vec2(coord, 6.0), 5, 2.0);
    c = textureProjOffset(sampler1D(tex1D, samp), vec4(coord, 0.0, 0.0, 6.0), 5, 2.0);
    c = texelFetch(sampler1D(tex1D, samp), int(coord), 3);
    c = texelFetchOffset(sampler1D(tex1D, samp), int(coord), 3, 5);
}

#if HAS_1D_DEPTH_TEXTURES
void testTex1DShadow(float coord) {
    int size1DShadow = textureSize(sampler1DShadow(tex1DShadow, sampShadow), 0);
    int levels = textureQueryLevels(sampler1DShadow(tex1DShadow, sampShadow));
    float d;
    d = texture(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0));
    // d = texture(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 2.0);
    d = textureGrad(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 4.0, 4.0);
    d = textureGradOffset(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 4.0, 4.0, 5);
    d = textureLod(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 3.0);
    d = textureLodOffset(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 3.0, 5);
    d = textureOffset(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 5);
    // d = textureOffset(sampler1DShadow(tex1DShadow, sampShadow), vec3(coord, 1.0, 1.0), 5, 2.0);
    d = textureProj(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0));
    // d = textureProj(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 2.0);
    d = textureProjGrad(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 4.0, 4.0);
    d = textureProjGradOffset(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 4.0, 4.0, 5);
    d = textureProjLod(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 3.0);
    d = textureProjLodOffset(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 3.0, 5);
    d = textureProjOffset(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 5);
    // d = textureProjOffset(sampler1DShadow(tex1DShadow, sampShadow), vec4(coord, 0.0, 1.0, 6.0), 5, 2.0);
}
#endif

void testTex1DArray(in vec2 coord) {
    ivec2 size1DArray = textureSize(sampler1DArray(tex1DArray, samp), 0);
    int levels = textureQueryLevels(sampler1DArray(tex1DArray, samp));
    vec4 c;
    c = texture(sampler1DArray(tex1DArray, samp), coord);
    c = texture(sampler1DArray(tex1DArray, samp), coord, 2.0);
    c = textureGrad(sampler1DArray(tex1DArray, samp), coord, 4.0, 4.0);
    c = textureGradOffset(sampler1DArray(tex1DArray, samp), coord, 4.0, 4.0, 5);
    c = textureLod(sampler1DArray(tex1DArray, samp), coord, 3.0);
    c = textureLodOffset(sampler1DArray(tex1DArray, samp), coord, 3.0, 5);
    c = textureOffset(sampler1DArray(tex1DArray, samp), coord, 5);
    c = textureOffset(sampler1DArray(tex1DArray, samp), coord, 5, 2.0);
    c = texelFetch(sampler1DArray(tex1DArray, samp), ivec2(coord), 3);
    c = texelFetchOffset(sampler1DArray(tex1DArray, samp), ivec2(coord), 3, 5);
}

#if HAS_1D_DEPTH_TEXTURES
void testTex1DArrayShadow(in vec2 coord) {
    ivec2 size1DArrayShadow = textureSize(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), 0);
    int levels = textureQueryLevels(sampler1DArrayShadow(tex1DArrayShadow, sampShadow));
    float d;
    d = texture(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0));
    d = textureGrad(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0), 4.0, 4.0);
    d = textureGradOffset(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0), 4.0, 4.0, 5);
    d = textureLod(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0), 3.0);
    d = textureLodOffset(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0), 3.0, 5);
    d = textureOffset(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0), 5);
    // d = textureOffset(sampler1DArrayShadow(tex1DArrayShadow, sampShadow), vec3(coord, 1.0), 5, 2.0);
}
#endif

void testTex2D(in vec2 coord) {
    ivec2 size2D = textureSize(sampler2D(tex2D, samp), 0);
    int levels = textureQueryLevels(sampler2D(tex2D, samp));
    vec4 c;
    c = texture(sampler2D(tex2D, samp), coord);
    c = texture(sampler2D(tex2D, samp), coord, 2.0);
    c = textureGrad(sampler2D(tex2D, samp), coord, vec2(4.0), vec2(4.0));
    c = textureGradOffset(sampler2D(tex2D, samp), coord, vec2(4.0), vec2(4.0), ivec2(5));
    c = textureLod(sampler2D(tex2D, samp), coord, 3.0);
    c = textureLodOffset(sampler2D(tex2D, samp), coord, 3.0, ivec2(5));
    c = textureOffset(sampler2D(tex2D, samp), coord, ivec2(5));
    c = textureOffset(sampler2D(tex2D, samp), coord, ivec2(5), 2.0);
    c = textureProj(sampler2D(tex2D, samp), vec3(coord, 6.0));
    c = textureProj(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0));
    c = textureProj(sampler2D(tex2D, samp), vec3(coord, 6.0), 2.0);
    c = textureProj(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), 2.0);
    c = textureProjGrad(sampler2D(tex2D, samp), vec3(coord, 6.0), vec2(4.0), vec2(4.0));
    c = textureProjGrad(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), vec2(4.0), vec2(4.0));
    c = textureProjGradOffset(sampler2D(tex2D, samp), vec3(coord, 6.0), vec2(4.0), vec2(4.0), ivec2(5));
    c = textureProjGradOffset(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), vec2(4.0), vec2(4.0), ivec2(5));
    c = textureProjLod(sampler2D(tex2D, samp), vec3(coord, 6.0), 3.0);
    c = textureProjLod(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), 3.0);
    c = textureProjLodOffset(sampler2D(tex2D, samp), vec3(coord, 6.0), 3.0, ivec2(5));
    c = textureProjLodOffset(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), 3.0, ivec2(5));
    c = textureProjOffset(sampler2D(tex2D, samp), vec3(coord, 6.0), ivec2(5));
    c = textureProjOffset(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), ivec2(5));
    c = textureProjOffset(sampler2D(tex2D, samp), vec3(coord, 6.0), ivec2(5), 2.0);
    c = textureProjOffset(sampler2D(tex2D, samp), vec4(coord, 0.0, 6.0), ivec2(5), 2.0);
    c = texelFetch(sampler2D(tex2D, samp), ivec2(coord), 3);
    c = texelFetchOffset(sampler2D(tex2D, samp), ivec2(coord), 3, ivec2(5));
}

void testTex2DShadow(vec2 coord) {
    ivec2 size2DShadow = textureSize(sampler2DShadow(tex2DShadow, sampShadow), 0);
    int levels = textureQueryLevels(sampler2DShadow(tex2DShadow, sampShadow));
    float d;
    d = texture(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0));
    // d = texture(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), 2.0);
    d = textureGrad(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), vec2(4.0), vec2(4.0));
    d = textureGradOffset(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), vec2(4.0), vec2(4.0), ivec2(5));
    d = textureLod(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), 3.0);
    d = textureLodOffset(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), 3.0, ivec2(5));
    d = textureOffset(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), ivec2(5));
    // d = textureOffset(sampler2DShadow(tex2DShadow, sampShadow), vec3(coord, 1.0), ivec2(5), 2.0);
    d = textureProj(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0));
    // d = textureProj(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), 2.0);
    d = textureProjGrad(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), vec2(4.0), vec2(4.0));
    d = textureProjGradOffset(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), vec2(4.0), vec2(4.0), ivec2(5));
    d = textureProjLod(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), 3.0);
    d = textureProjLodOffset(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), 3.0, ivec2(5));
    d = textureProjOffset(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), ivec2(5));
    // d = textureProjOffset(sampler2DShadow(tex2DShadow, sampShadow), vec4(coord, 1.0, 6.0), ivec2(5), 2.0);
}

void testTex2DArray(in vec3 coord) {
    ivec3 size2DArray = textureSize(sampler2DArray(tex2DArray, samp), 0);
    int levels = textureQueryLevels(sampler2DArray(tex2DArray, samp));
    vec4 c;
    c = texture(sampler2DArray(tex2DArray, samp), coord);
    c = texture(sampler2DArray(tex2DArray, samp), coord, 2.0);
    c = textureGrad(sampler2DArray(tex2DArray, samp), coord, vec2(4.0), vec2(4.0));
    c = textureGradOffset(sampler2DArray(tex2DArray, samp), coord, vec2(4.0), vec2(4.0), ivec2(5));
    c = textureLod(sampler2DArray(tex2DArray, samp), coord, 3.0);
    c = textureLodOffset(sampler2DArray(tex2DArray, samp), coord, 3.0, ivec2(5));
    c = textureOffset(sampler2DArray(tex2DArray, samp), coord, ivec2(5));
    c = textureOffset(sampler2DArray(tex2DArray, samp), coord, ivec2(5), 2.0);
    c = texelFetch(sampler2DArray(tex2DArray, samp), ivec3(coord), 3);
    c = texelFetchOffset(sampler2DArray(tex2DArray, samp), ivec3(coord), 3, ivec2(5));
}

void testTex2DArrayShadow(in vec3 coord) {
    ivec3 size2DArrayShadow = textureSize(sampler2DArrayShadow(tex2DArrayShadow, sampShadow), 0);
    int levels = textureQueryLevels(sampler2DArrayShadow(tex2DArrayShadow, sampShadow));
    float d;
    d = texture(sampler2DArrayShadow(tex2DArrayShadow, sampShadow), vec4(coord, 1.0));
    d = textureGrad(sampler2DArrayShadow(tex2DArrayShadow, sampShadow), vec4(coord, 1.0), vec2(4.0), vec2(4.0));
    d = textureGradOffset(sampler2DArrayShadow(tex2DArrayShadow, sampShadow), vec4(coord, 1.0), vec2(4.0), vec2(4.0), ivec2(5));
    d = textureOffset(sampler2DArrayShadow(tex2DArrayShadow, sampShadow), vec4(coord, 1.0), ivec2(5));
}

void testTexCube(in vec3 coord) {
    ivec2 sizeCube = textureSize(samplerCube(texCube, samp), 0);
    int levels = textureQueryLevels(samplerCube(texCube, samp));
    vec4 c;
    c = texture(samplerCube(texCube, samp), coord);
    c = texture(samplerCube(texCube, samp), coord, 2.0);
    c = textureGrad(samplerCube(texCube, samp), coord, vec3(4.0), vec3(4.0));
    c = textureLod(samplerCube(texCube, samp), coord, 3.0);
}

void testTexCubeShadow(in vec3 coord) {
    ivec2 sizeCubeShadow = textureSize(samplerCubeShadow(texCubeShadow, sampShadow), 0);
    int levels = textureQueryLevels(samplerCubeShadow(texCubeShadow, sampShadow));
    float d;
    d = texture(samplerCubeShadow(texCubeShadow, sampShadow), vec4(coord, 1.0));
    d = textureGrad(samplerCubeShadow(texCubeShadow, sampShadow), vec4(coord, 1.0), vec3(4.0), vec3(4.0));
}

void testTexCubeArray(in vec4 coord) {
    ivec3 sizeCubeArray = textureSize(samplerCubeArray(texCubeArray, samp), 0);
    int levels = textureQueryLevels(samplerCubeArray(texCubeArray, samp));
    vec4 c;
    c = texture(samplerCubeArray(texCubeArray, samp), coord);
    c = texture(samplerCubeArray(texCubeArray, samp), coord, 2.0);
    c = textureGrad(samplerCubeArray(texCubeArray, samp), coord, vec3(4.0), vec3(4.0));
    c = textureLod(samplerCubeArray(texCubeArray, samp), coord, 3.0);
}

void testTexCubeArrayShadow(in vec4 coord) {
    ivec3 sizeCubeArrayShadow = textureSize(samplerCubeArrayShadow(texCubeArrayShadow, sampShadow), 0);
    int levels = textureQueryLevels(samplerCubeArrayShadow(texCubeArrayShadow, sampShadow));
    float d;
    d = texture(samplerCubeArrayShadow(texCubeArrayShadow, sampShadow), coord, 1.0);
    // The rest of the variants aren't defined by GLSL.
}

void testTex3D(in vec3 coord) {
    ivec3 size3D = textureSize(sampler3D(tex3D, samp), 0);
    int levels = textureQueryLevels(sampler3D(tex3D, samp));
    vec4 c;
    c = texture(sampler3D(tex3D, samp), coord);
    c = texture(sampler3D(tex3D, samp), coord, 2.0);
    c = textureProj(sampler3D(tex3D, samp), vec4(coord, 6.0));
    c = textureProj(sampler3D(tex3D, samp), vec4(coord, 6.0), 2.0);
    c = textureProjOffset(sampler3D(tex3D, samp), vec4(coord, 6.0), ivec3(5));
    c = textureProjOffset(sampler3D(tex3D, samp), vec4(coord, 6.0), ivec3(5), 2.0);
    c = textureProjLod(sampler3D(tex3D, samp), vec4(coord, 6.0), 3.0);
    c = textureProjLodOffset(sampler3D(tex3D, samp), vec4(coord, 6.0), 3.0, ivec3(5));
    c = textureProjGrad(sampler3D(tex3D, samp), vec4(coord, 6.0), vec3(4.0), vec3(4.0));
    c = textureProjGradOffset(sampler3D(tex3D, samp), vec4(coord, 6.0), vec3(4.0), vec3(4.0), ivec3(5));
    c = textureGrad(sampler3D(tex3D, samp), coord, vec3(4.0), vec3(4.0));
    c = textureGradOffset(sampler3D(tex3D, samp), coord, vec3(4.0), vec3(4.0), ivec3(5));
    c = textureLod(sampler3D(tex3D, samp), coord, 3.0);
    c = textureLodOffset(sampler3D(tex3D, samp), coord, 3.0, ivec3(5));
    c = textureOffset(sampler3D(tex3D, samp), coord, ivec3(5));
    c = textureOffset(sampler3D(tex3D, samp), coord, ivec3(5), 2.0);
    c = texelFetch(sampler3D(tex3D, samp), ivec3(coord), 3);
    c = texelFetchOffset(sampler3D(tex3D, samp), ivec3(coord), 3, ivec3(5));
}

void testTex2DMS(in vec2 coord) {
    ivec2 size2DMS = textureSize(sampler2DMS(tex2DMS, samp));
    vec4 c;
    c = texelFetch(sampler2DMS(tex2DMS, samp), ivec2(coord), 3);
}

void testTex2DMSArray(in vec3 coord) {
    ivec3 size2DMSArray = textureSize(sampler2DMSArray(tex2DMSArray, samp));
    vec4 c;
    c = texelFetch(sampler2DMSArray(tex2DMSArray, samp), ivec3(coord), 3);
}

void main() {}
