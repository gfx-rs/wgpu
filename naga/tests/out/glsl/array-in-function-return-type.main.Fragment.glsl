#version 310 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

float[2] ret_array() {
    return float[2](1.0, 2.0);
}

void main() {
    float _e0[2] = ret_array();
    _fs2p_location0 = vec4(_e0[0], _e0[1], 0.0, 1.0);
    return;
}

