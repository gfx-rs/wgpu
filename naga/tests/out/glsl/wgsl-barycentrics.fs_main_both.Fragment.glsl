#version 450 core
#extension GL_EXT_fragment_shader_barycentric : require
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec3 bary_2 = gl_BaryCoordEXT;
    vec3 bary_no_persp = gl_BaryCoordNoPerspEXT;
    _fs2p_location0 = vec4(bary_2.xy, bary_no_persp.z, 1.0);
    return;
}

