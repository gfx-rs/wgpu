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
    vec2 a_1 = (((vec2(1.0) + vec2(2.0)) - vec2(3.0)) / vec2(4.0));
    ivec4 b = (ivec4(5) % ivec4(2));
    return (a_1.xyxy + vec4(b));
}

vec3 bool_cast(vec3 x) {
    bvec3 y = bvec3(x);
    return vec3(y);
}

float constructors() {
    Foo foo = Foo(vec4(0.0), 0);
    bool unnamed = false;
    int unnamed_1 = 0;
    uint unnamed_2 = 0u;
    float unnamed_3 = 0.0;
    uvec2 unnamed_4 = uvec2(0u, 0u);
    mat2x2 unnamed_5 = mat2x2(vec2(0.0, 0.0), vec2(0.0, 0.0));
    Foo unnamed_6[3] = Foo[3](Foo(vec4(0.0, 0.0, 0.0, 0.0), 0), Foo(vec4(0.0, 0.0, 0.0, 0.0), 0), Foo(vec4(0.0, 0.0, 0.0, 0.0), 0));
    Foo unnamed_7 = Foo(vec4(0.0, 0.0, 0.0, 0.0), 0);
    uvec2 unnamed_8 = uvec2(0u);
    mat2x2 unnamed_9 = mat2x2(0.0);
    int unnamed_10[4] = int[4](0, 0, 0, 0);
    foo = Foo(vec4(1.0), 1);
    mat2x2 mat2comp = mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    mat4x4 mat4comp = mat4x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
    unnamed_8 = uvec2(0u);
    unnamed_9 = mat2x2(vec2(0.0), vec2(0.0));
    unnamed_10 = int[4](0, 1, 2, 3);
    float _e70 = foo.a.x;
    return _e70;
}

void logical() {
    bool unnamed_11 = (! true);
    bool unnamed_12 = (true || false);
    bool unnamed_13 = (true && false);
    bool unnamed_14 = (true || false);
    bvec3 unnamed_15 = (bvec3(true) | bvec3(false));
    bool unnamed_16 = (true && false);
    bvec4 unnamed_17 = (bvec4(true) & bvec4(false));
}

void arithmetic() {
    ivec2 unnamed_18 = (- ivec2(1));
    vec2 unnamed_19 = (- vec2(1.0));
    int unnamed_20 = (2 + 1);
    uint unnamed_21 = (2u + 1u);
    float unnamed_22 = (2.0 + 1.0);
    ivec2 unnamed_23 = (ivec2(2) + ivec2(1));
    uvec3 unnamed_24 = (uvec3(2u) + uvec3(1u));
    vec4 unnamed_25 = (vec4(2.0) + vec4(1.0));
    int unnamed_26 = (2 - 1);
    uint unnamed_27 = (2u - 1u);
    float unnamed_28 = (2.0 - 1.0);
    ivec2 unnamed_29 = (ivec2(2) - ivec2(1));
    uvec3 unnamed_30 = (uvec3(2u) - uvec3(1u));
    vec4 unnamed_31 = (vec4(2.0) - vec4(1.0));
    int unnamed_32 = (2 * 1);
    uint unnamed_33 = (2u * 1u);
    float unnamed_34 = (2.0 * 1.0);
    ivec2 unnamed_35 = (ivec2(2) * ivec2(1));
    uvec3 unnamed_36 = (uvec3(2u) * uvec3(1u));
    vec4 unnamed_37 = (vec4(2.0) * vec4(1.0));
    int unnamed_38 = (2 / 1);
    uint unnamed_39 = (2u / 1u);
    float unnamed_40 = (2.0 / 1.0);
    ivec2 unnamed_41 = (ivec2(2) / ivec2(1));
    uvec3 unnamed_42 = (uvec3(2u) / uvec3(1u));
    vec4 unnamed_43 = (vec4(2.0) / vec4(1.0));
    int unnamed_44 = (2 % 1);
    uint unnamed_45 = (2u % 1u);
    float unnamed_46 = (2.0 - 1.0 * trunc(2.0 / 1.0));
    ivec2 unnamed_47 = (ivec2(2) % ivec2(1));
    uvec3 unnamed_48 = (uvec3(2u) % uvec3(1u));
    vec4 unnamed_49 = (vec4(2.0) - vec4(1.0) * trunc(vec4(2.0) / vec4(1.0)));
    ivec2 unnamed_50 = (ivec2(2) + ivec2(1));
    ivec2 unnamed_51 = (ivec2(2) + ivec2(1));
    ivec2 unnamed_52 = (ivec2(2) - ivec2(1));
    ivec2 unnamed_53 = (ivec2(2) - ivec2(1));
    ivec2 unnamed_54 = (ivec2(2) * 1);
    ivec2 unnamed_55 = (2 * ivec2(1));
    ivec2 unnamed_56 = (ivec2(2) / ivec2(1));
    ivec2 unnamed_57 = (ivec2(2) / ivec2(1));
    ivec2 unnamed_58 = (ivec2(2) % ivec2(1));
    ivec2 unnamed_59 = (ivec2(2) % ivec2(1));
    mat3x3 unnamed_60 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * 1.0);
    mat3x3 unnamed_61 = (2.0 * mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    vec3 unnamed_62 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * vec4(1.0));
    vec4 unnamed_63 = (vec3(2.0) * mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_64 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * mat3x4(vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0)));
}

void bit() {
    int unnamed_65 = (~ 1);
    uint unnamed_66 = (~ 1u);
    int unnamed_67 = (2 | 1);
    uint unnamed_68 = (2u | 1u);
    ivec2 unnamed_69 = (ivec2(2) | ivec2(1));
    uvec3 unnamed_70 = (uvec3(2u) | uvec3(1u));
    int unnamed_71 = (2 & 1);
    uint unnamed_72 = (2u & 1u);
    ivec2 unnamed_73 = (ivec2(2) & ivec2(1));
    uvec3 unnamed_74 = (uvec3(2u) & uvec3(1u));
    int unnamed_75 = (2 ^ 1);
    uint unnamed_76 = (2u ^ 1u);
    ivec2 unnamed_77 = (ivec2(2) ^ ivec2(1));
    uvec3 unnamed_78 = (uvec3(2u) ^ uvec3(1u));
    int unnamed_79 = (2 << 1u);
    uint unnamed_80 = (2u << 1u);
    ivec2 unnamed_81 = (ivec2(2) << uvec2(1u));
    uvec3 unnamed_82 = (uvec3(2u) << uvec3(1u));
    int unnamed_83 = (2 >> 1u);
    uint unnamed_84 = (2u >> 1u);
    ivec2 unnamed_85 = (ivec2(2) >> uvec2(1u));
    uvec3 unnamed_86 = (uvec3(2u) >> uvec3(1u));
}

void comparison() {
    bool unnamed_87 = (2 == 1);
    bool unnamed_88 = (2u == 1u);
    bool unnamed_89 = (2.0 == 1.0);
    bvec2 unnamed_90 = equal(ivec2(2), ivec2(1));
    bvec3 unnamed_91 = equal(uvec3(2u), uvec3(1u));
    bvec4 unnamed_92 = equal(vec4(2.0), vec4(1.0));
    bool unnamed_93 = (2 != 1);
    bool unnamed_94 = (2u != 1u);
    bool unnamed_95 = (2.0 != 1.0);
    bvec2 unnamed_96 = notEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_97 = notEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_98 = notEqual(vec4(2.0), vec4(1.0));
    bool unnamed_99 = (2 < 1);
    bool unnamed_100 = (2u < 1u);
    bool unnamed_101 = (2.0 < 1.0);
    bvec2 unnamed_102 = lessThan(ivec2(2), ivec2(1));
    bvec3 unnamed_103 = lessThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_104 = lessThan(vec4(2.0), vec4(1.0));
    bool unnamed_105 = (2 <= 1);
    bool unnamed_106 = (2u <= 1u);
    bool unnamed_107 = (2.0 <= 1.0);
    bvec2 unnamed_108 = lessThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_109 = lessThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_110 = lessThanEqual(vec4(2.0), vec4(1.0));
    bool unnamed_111 = (2 > 1);
    bool unnamed_112 = (2u > 1u);
    bool unnamed_113 = (2.0 > 1.0);
    bvec2 unnamed_114 = greaterThan(ivec2(2), ivec2(1));
    bvec3 unnamed_115 = greaterThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_116 = greaterThan(vec4(2.0), vec4(1.0));
    bool unnamed_117 = (2 >= 1);
    bool unnamed_118 = (2u >= 1u);
    bool unnamed_119 = (2.0 >= 1.0);
    bvec2 unnamed_120 = greaterThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_121 = greaterThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_122 = greaterThanEqual(vec4(2.0), vec4(1.0));
}

void assignment() {
    int a = 1;
    int _e6 = a;
    a = (_e6 + 1);
    int _e9 = a;
    a = (_e9 - 1);
    int _e12 = a;
    int _e13 = a;
    a = (_e12 * _e13);
    int _e15 = a;
    int _e16 = a;
    a = (_e15 / _e16);
    int _e18 = a;
    a = (_e18 % 1);
    int _e21 = a;
    a = (_e21 & 0);
    int _e24 = a;
    a = (_e24 | 0);
    int _e27 = a;
    a = (_e27 ^ 0);
    int _e30 = a;
    a = (_e30 << 2u);
    int _e33 = a;
    a = (_e33 >> 1u);
    int _e36 = a;
    a = (_e36 + 1);
    int _e39 = a;
    a = (_e39 - 1);
    return;
}

void main() {
    vec4 _e4 = builtins();
    vec4 _e5 = splat();
    vec3 _e7 = bool_cast(vec4(1.0, 1.0, 1.0, 1.0).xyz);
    float _e8 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}

