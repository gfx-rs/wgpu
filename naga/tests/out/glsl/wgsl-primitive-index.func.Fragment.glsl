#version 310 es
#extension GL_OES_geometry_shader : require

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    uint index = uint(gl_PrimitiveID);
    _fs2p_location0 = vec4(float(index), 1.0, 1.0, 1.0);
    return;
}

