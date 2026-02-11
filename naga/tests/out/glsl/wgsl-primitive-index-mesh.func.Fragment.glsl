#version 320 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    uint index = uint(gl_PrimitiveID);
    _fs2p_location0 = vec4(float(index), 1.0, 1.0, 1.0);
    return;
}

