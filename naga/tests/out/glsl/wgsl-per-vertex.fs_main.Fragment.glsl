#version 450 core
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) pervertexEXT in float _vs2fs_location0;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    float v = _vs2fs_location0;
    _fs2p_location0 = vec4(v[0], v[1], v[2], 1.0);
    return;
}

