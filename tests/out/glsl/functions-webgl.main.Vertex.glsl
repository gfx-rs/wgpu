#version 320 es

precision highp float;
precision highp int;


vec2 test_fma() {
    vec2 a = vec2(2.0, 2.0);
    vec2 b = vec2(0.5, 0.5);
    vec2 c = vec2(0.5, 0.5);
    return fma(a, b, c);
}

void main() {
    vec2 _e0 = test_fma();
    return;
}

