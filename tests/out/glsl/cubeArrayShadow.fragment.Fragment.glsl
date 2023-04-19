#version 310 es
#extension GL_EXT_texture_cube_map_array : require

precision highp float;
precision highp int;

uniform highp samplerCubeArrayShadow _group_0_binding_4_fs;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec3 frag_ls = vec3(1.0, 1.0, 2.0);
    float a = texture(_group_0_binding_4_fs, vec4(frag_ls, 1), 1.0);
    _fs2p_location0 = vec4(a, 1.0, 1.0, 1.0);
    return;
}

