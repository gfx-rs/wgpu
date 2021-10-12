#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Foo {
    vec4 a;
    int b;
};


vec4 builtins() {
    int s1_ = (true ? 1 : 0);
    vec4 s2_ = (true ? vec4(1.0, 1.0, 1.0, 1.0) : vec4(0.0, 0.0, 0.0, 0.0));
    vec4 s3_ = mix(vec4(1.0, 1.0, 1.0, 1.0), vec4(0.0, 0.0, 0.0, 0.0), bvec4(false, false, false, false));
    vec4 m1_ = mix(vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0), vec4(0.5, 0.5, 0.5, 0.5));
    vec4 m2_ = mix(vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0), 0.10000000149011612);
    float b1_ = intBitsToFloat(ivec4(1, 1, 1, 1).x);
    vec4 b2_ = intBitsToFloat(ivec4(1, 1, 1, 1));
    ivec4 v_i32_zero = ivec4(vec4(0.0, 0.0, 0.0, 0.0));
    return (((((vec4((ivec4(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4(b1_)) + b2_);
}

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

vec3 bool_cast(vec3 x) {
    bvec3 y = bvec3(x);
    return vec3(y);
}

float constructors() {
    Foo foo;
    foo = Foo(vec4(1.0), 1);
    float _e11 = foo.a.x;
    return _e11;
}

void modulo() {
    int a1 = (1 % 1);
    float b1 = (1.0 - 1.0 * trunc(1.0 / 1.0));
    ivec3 c = (ivec3(1) % ivec3(1));
    vec3 d = (vec3(1.0) - vec3(1.0) * trunc(vec3(1.0) / vec3(1.0)));
}

void main() {
    vec4 _e4 = builtins();
    vec4 _e5 = splat();
    int _e6 = unary();
    vec3 _e8 = bool_cast(vec4(1.0, 1.0, 1.0, 1.0).xyz);
    float _e9 = constructors();
    modulo();
    return;
}

