#version 310 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

void derivatives() {
    float x = dFdx(0.0);
    float y = dFdy(0.0);
    float width = fwidth(0.0);
}

void main() {
    derivatives();
    _fs2p_location0 = vec4(0.0);
    return;
}

