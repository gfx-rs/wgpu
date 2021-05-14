#version 310 es

precision highp float;

layout(location = 0) out vec4 _fs2p_location0;

void main() {
    vec4 foo = gl_FragCoord;
    vec4 _expr1 = dFdx(foo);
    vec4 _expr2 = dFdy(foo);
    vec4 _expr3 = fwidth(foo);
    _fs2p_location0 = ((_expr1 + _expr2) * _expr3);
    return;
}

