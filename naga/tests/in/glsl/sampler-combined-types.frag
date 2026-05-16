#version 450 core

// Exercises combined image-sampler type names (`sampler2D` etc.) as direct
// uniform variable types with explicit descriptor-set / binding layout qualifiers.
// This regressed as NotImplemented("variable qualifier") before the fix.

layout(set = 0, binding = 0) uniform sampler2D u_sampled;
layout(set = 0, binding = 1) uniform isampler2D u_isampled;
layout(set = 0, binding = 2) uniform usampler3D u_usampled_3d;
layout(set = 0, binding = 3) uniform samplerCube u_cube;
layout(set = 0, binding = 4) uniform sampler2DArray u_array;
layout(set = 0, binding = 5) uniform sampler2DShadow u_shadow;

layout(set = 1, binding = 0) uniform sampler samp;

void main() {
    ivec2 coord2 = ivec2(gl_FragCoord.xy);
    vec3 coord3 = vec3(gl_FragCoord.xy, 0.0);

    vec4  c0 = texelFetch(u_sampled, coord2, 0);
    ivec4 c1 = texelFetch(u_isampled, coord2, 0);
    uvec4 c2 = texelFetch(u_usampled_3d, ivec3(coord2, 0), 0);
    vec4  c3 = texture(samplerCube(u_cube, samp), coord3);
    vec4  c4 = texture(sampler2DArray(u_array, samp), coord3);

    gl_FragDepth = c0.r + float(c1.r) + float(c2.r) + c3.r + c4.r;
}
