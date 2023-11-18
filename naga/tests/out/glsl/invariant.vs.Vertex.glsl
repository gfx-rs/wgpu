#version 300 es

precision highp float;
precision highp int;

uniform uint _naga_vs_base_instance;

invariant gl_Position;

void main() {
    gl_Position = vec4(0.0);
    return;
}

