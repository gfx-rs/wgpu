#version 310 es

precision highp float;

layout(location = 0) out vec4 _fs2p_location0;

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

int unary1() {
    if ((! true)) {
        return 1;
    } else {
        return (~ 1);
    }
}

void main() {
    vec4 vector1_ = vec4(1.0);
    vec4 vector2_ = vec4(1.0);
    _fs2p_location0 = (true ? vector1_ : vector2_);
    return;
}

