#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


vec4 splat() {
    vec2 a = (((vec2(1.0) + vec2(2.0)) - vec2(3.0)) / vec2(4.0));
    ivec4 b = (ivec4(5) % ivec4(2));
    return (a.xyxy + vec4(b));
}

int unary() {
    if ((! true)) {
        return 1;
    } else {
        return (~ 1);
    }
}

vec4 selection() {
    vec4 vector1_ = vec4(1.0);
    vec4 vector2_ = vec4(1.0);
    int a = (true ? 1 : 0);
    return (true ? vector2_ : vector1_);
}

void main() {
    vec4 _expr0 = splat();
    int _expr1 = unary();
    vec4 _expr2 = selection();
    return;
}

