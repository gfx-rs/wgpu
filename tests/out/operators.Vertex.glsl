#version 310 es

precision highp float;


void main() {
    vec2 _expr10 = (((vec2(1.0) + vec2(2.0)) - vec2(3.0)) / vec2(4.0));
    gl_Position = (vec4(_expr10[0], _expr10[1], _expr10[0], _expr10[1]) + vec4((ivec4(5) % ivec4(2))));
    return;
}

