#version 310 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

float[2] ret_array() {
    return float[2](1.0, 2.0);
}

float[3][2] ret_array_array() {
    float _e0[2] = ret_array();
    float _e1[2] = ret_array();
    float _e2[2] = ret_array();
    return float[3][2](_e0, _e1, _e2);
}

void main() {
    float _e0[3][2] = ret_array_array();
    _fs2p_location0 = vec4(_e0[0][0], _e0[0][1], 0.0, 1.0);
    return;
}

