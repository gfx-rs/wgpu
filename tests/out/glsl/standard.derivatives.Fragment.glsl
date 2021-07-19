#version 310 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 foo = gl_FragCoord;
    vec4 x = dFdx(foo);
    vec4 y = dFdy(foo);
    vec4 z = fwidth(foo);
    _fs2p_location0 = ((x + y) * z);
    return;
}

