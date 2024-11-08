#version 430 core
uniform sampler2D _group_0_binding_1_fs;

uniform usampler2D _group_0_binding_2_fs;

uniform isampler2D _group_0_binding_3_fs;

uniform sampler2DShadow _group_1_binding_2_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec2 tc = vec2(0.5);
    vec4 s2d = textureGather(_group_0_binding_1_fs, vec2(tc), 1);
    vec4 s2d_offset = textureGatherOffset(_group_0_binding_1_fs, vec2(tc), ivec2(3, 1), 3);
    vec4 s2d_depth = textureGather(_group_1_binding_2_fs, vec2(tc), 0.5);
    vec4 s2d_depth_offset = textureGatherOffset(_group_1_binding_2_fs, vec2(tc), 0.5, ivec2(3, 1));
    uvec4 u = textureGather(_group_0_binding_2_fs, vec2(tc), 0);
    ivec4 i = textureGather(_group_0_binding_3_fs, vec2(tc), 0);
    vec4 f = (vec4(u) + vec4(i));
    _fs2p_location0 = ((((s2d + s2d_offset) + s2d_depth) + s2d_depth_offset) + f);
    return;
}

