#version 450 core
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec3 bary_1 = gl_BaryCoordNoPerspEXT;
    _fs2p_location0 = vec4(bary_1, 1.0);
    return;
}

