#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Foo {
    vec4 a;
    int b;
};
const vec3 const2_ = vec3(0.0, 1.0, 2.0);
const mat2x2 const3_ = mat2x2(vec2(0.0, 1.0), vec2(2.0, 3.0));
const mat2x2 const4_[1] = mat2x2[1](mat2x2(vec2(0.0, 1.0), vec2(2.0, 3.0)));
const bool cz0_ = false;
const int cz1_ = 0;
const uint cz2_ = 0u;
const float cz3_ = 0.0;
const uvec2 cz4_ = uvec2(0u);
const mat2x2 cz5_ = mat2x2(0.0);
const Foo cz6_[3] = Foo[3](Foo(vec4(0.0), 0), Foo(vec4(0.0), 0), Foo(vec4(0.0), 0));
const Foo cz7_ = Foo(vec4(0.0), 0);
const int cp3_[4] = int[4](0, 1, 2, 3);


void main() {
    Foo foo = Foo(vec4(0.0), 0);
    foo = Foo(vec4(1.0), 1);
    mat2x2 m0_ = mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    mat4x4 m1_ = mat4x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
    uvec2 cit0_ = uvec2(0u);
    mat2x2 cit1_ = mat2x2(vec2(0.0), vec2(0.0));
    int cit2_[4] = int[4](0, 1, 2, 3);
    bool ic0_ = bool(false);
    int ic1_ = int(0);
    uint ic2_ = uint(0u);
    float ic3_ = float(0.0);
    uvec2 ic4_ = uvec2(uvec2(0u));
    mat2x3 ic5_ = mat2x3(mat2x3(0.0));
    uvec2 ic6_ = uvec2(0u);
    mat2x3 ic7_ = mat2x3(0.0);
}

