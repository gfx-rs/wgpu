#version 310 es

precision highp float;


void main() {
    gl_Position = ((((vec2(1.0) + vec2(2.0)) - vec2(3.0)) / vec2(4.0)).xyxy + vec4((ivec4(5) % ivec4(2))));
    return;
}

