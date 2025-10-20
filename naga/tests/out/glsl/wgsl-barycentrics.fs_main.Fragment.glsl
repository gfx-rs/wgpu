#version 310 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec3 bary = gl_BaryCoordEXT;
    _fs2p_location0 = vec4(bary, 1.0);
    return;
}

