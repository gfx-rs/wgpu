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
    bvec2 unnamed_12 = (! bvec2(true));
    bool unnamed_13 = (true || false);
    bool unnamed_14 = (true && false);
    bool unnamed_15 = (true || false);
    bvec3 unnamed_16 = (bvec3(true) | bvec3(false));
    bool unnamed_17 = (true && false);
    bvec4 unnamed_18 = (bvec4(true) & bvec4(false));
}

void arithmetic() {
    ivec2 unnamed_19 = (- ivec2(1));
    vec2 unnamed_20 = (- vec2(1.0));
    int unnamed_21 = (2 + 1);
    uint unnamed_22 = (2u + 1u);
    float unnamed_23 = (2.0 + 1.0);
    ivec2 unnamed_24 = (ivec2(2) + ivec2(1));
    uvec3 unnamed_25 = (uvec3(2u) + uvec3(1u));
    vec4 unnamed_26 = (vec4(2.0) + vec4(1.0));
    int unnamed_27 = (2 - 1);
    uint unnamed_28 = (2u - 1u);
    float unnamed_29 = (2.0 - 1.0);
    ivec2 unnamed_30 = (ivec2(2) - ivec2(1));
    uvec3 unnamed_31 = (uvec3(2u) - uvec3(1u));
    vec4 unnamed_32 = (vec4(2.0) - vec4(1.0));
    int unnamed_33 = (2 * 1);
    uint unnamed_34 = (2u * 1u);
    float unnamed_35 = (2.0 * 1.0);
    ivec2 unnamed_36 = (ivec2(2) * ivec2(1));
    uvec3 unnamed_37 = (uvec3(2u) * uvec3(1u));
    vec4 unnamed_38 = (vec4(2.0) * vec4(1.0));
    int unnamed_39 = (2 / 1);
    uint unnamed_40 = (2u / 1u);
    float unnamed_41 = (2.0 / 1.0);
    ivec2 unnamed_42 = (ivec2(2) / ivec2(1));
    uvec3 unnamed_43 = (uvec3(2u) / uvec3(1u));
    vec4 unnamed_44 = (vec4(2.0) / vec4(1.0));
    int unnamed_45 = (2 % 1);
    uint unnamed_46 = (2u % 1u);
    float unnamed_47 = (2.0 - 1.0 * trunc(2.0 / 1.0));
    ivec2 unnamed_48 = (ivec2(2) % ivec2(1));
    uvec3 unnamed_49 = (uvec3(2u) % uvec3(1u));
    vec4 unnamed_50 = (vec4(2.0) - vec4(1.0) * trunc(vec4(2.0) / vec4(1.0)));
    ivec2 unnamed_51 = (ivec2(2) + ivec2(1));
    ivec2 unnamed_52 = (ivec2(2) + ivec2(1));
    ivec2 unnamed_53 = (ivec2(2) - ivec2(1));
    ivec2 unnamed_54 = (ivec2(2) - ivec2(1));
    ivec2 unnamed_55 = (ivec2(2) * 1);
    ivec2 unnamed_56 = (2 * ivec2(1));
    ivec2 unnamed_57 = (ivec2(2) / ivec2(1));
    ivec2 unnamed_58 = (ivec2(2) / ivec2(1));
    ivec2 unnamed_59 = (ivec2(2) % ivec2(1));
    ivec2 unnamed_60 = (ivec2(2) % ivec2(1));
    mat3x3 unnamed_61 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * 1.0);
    mat3x3 unnamed_62 = (2.0 * mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    vec3 unnamed_63 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * vec4(1.0));
    vec4 unnamed_64 = (vec3(2.0) * mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_65 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * mat3x4(vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0)));
}

void bit() {
    int unnamed_66 = (~ 1);
    uint unnamed_67 = (~ 1u);
    ivec2 unnamed_68 = (~ ivec2(1));
    uvec3 unnamed_69 = (~ uvec3(1u));
    int unnamed_70 = (2 | 1);
    uint unnamed_71 = (2u | 1u);
    ivec2 unnamed_72 = (ivec2(2) | ivec2(1));
    uvec3 unnamed_73 = (uvec3(2u) | uvec3(1u));
    int unnamed_74 = (2 & 1);
    uint unnamed_75 = (2u & 1u);
    ivec2 unnamed_76 = (ivec2(2) & ivec2(1));
    uvec3 unnamed_77 = (uvec3(2u) & uvec3(1u));
    int unnamed_78 = (2 ^ 1);
    uint unnamed_79 = (2u ^ 1u);
    ivec2 unnamed_80 = (ivec2(2) ^ ivec2(1));
    uvec3 unnamed_81 = (uvec3(2u) ^ uvec3(1u));
    int unnamed_82 = (2 << 1u);
    uint unnamed_83 = (2u << 1u);
    ivec2 unnamed_84 = (ivec2(2) << uvec2(1u));
    uvec3 unnamed_85 = (uvec3(2u) << uvec3(1u));
    int unnamed_86 = (2 >> 1u);
    uint unnamed_87 = (2u >> 1u);
    ivec2 unnamed_88 = (ivec2(2) >> uvec2(1u));
    uvec3 unnamed_89 = (uvec3(2u) >> uvec3(1u));
}

void comparison() {
    bool unnamed_90 = (2 == 1);
    bool unnamed_91 = (2u == 1u);
    bool unnamed_92 = (2.0 == 1.0);
    bvec2 unnamed_93 = equal(ivec2(2), ivec2(1));
    bvec3 unnamed_94 = equal(uvec3(2u), uvec3(1u));
    bvec4 unnamed_95 = equal(vec4(2.0), vec4(1.0));
    bool unnamed_96 = (2 != 1);
    bool unnamed_97 = (2u != 1u);
    bool unnamed_98 = (2.0 != 1.0);
    bvec2 unnamed_99 = notEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_100 = notEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_101 = notEqual(vec4(2.0), vec4(1.0));
    bool unnamed_102 = (2 < 1);
    bool unnamed_103 = (2u < 1u);
    bool unnamed_104 = (2.0 < 1.0);
    bvec2 unnamed_105 = lessThan(ivec2(2), ivec2(1));
    bvec3 unnamed_106 = lessThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_107 = lessThan(vec4(2.0), vec4(1.0));
    bool unnamed_108 = (2 <= 1);
    bool unnamed_109 = (2u <= 1u);
    bool unnamed_110 = (2.0 <= 1.0);
    bvec2 unnamed_111 = lessThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_112 = lessThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_113 = lessThanEqual(vec4(2.0), vec4(1.0));
    bool unnamed_114 = (2 > 1);
    bool unnamed_115 = (2u > 1u);
    bool unnamed_116 = (2.0 > 1.0);
    bvec2 unnamed_117 = greaterThan(ivec2(2), ivec2(1));
    bvec3 unnamed_118 = greaterThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_119 = greaterThan(vec4(2.0), vec4(1.0));
    bool unnamed_120 = (2 >= 1);
    bool unnamed_121 = (2u >= 1u);
    bool unnamed_122 = (2.0 >= 1.0);
    bvec2 unnamed_123 = greaterThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_124 = greaterThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_125 = greaterThanEqual(vec4(2.0), vec4(1.0));
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

