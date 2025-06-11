#version 310 es

precision highp float;
precision highp int;

layout(early_fragment_tests) in;
layout(location = 0) out vec4 _fs2p_location0;

void main() {
    _fs2p_location0 = vec4(0.4, 0.3, 0.2, 0.1);
    return;
}

