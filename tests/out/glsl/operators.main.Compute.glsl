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
    bvec2 unnamed_12 = not(bvec2(true));
    bool unnamed_13 = (true || false);
    bool unnamed_14 = (true && false);
    bool unnamed_15 = (true || false);
    bvec3 unnamed_16 = bvec3(bvec3(true).x || bvec3(false).x, bvec3(true).y || bvec3(false).y, bvec3(true).z || bvec3(false).z);
    bool unnamed_17 = (true && false);
    bvec4 unnamed_18 = bvec4(bvec4(true).x && bvec4(false).x, bvec4(true).y && bvec4(false).y, bvec4(true).z && bvec4(false).z, bvec4(true).w && bvec4(false).w);
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
    uvec2 unnamed_53 = (uvec2(2u) + uvec2(1u));
    uvec2 unnamed_54 = (uvec2(2u) + uvec2(1u));
    vec2 unnamed_55 = (vec2(2.0) + vec2(1.0));
    vec2 unnamed_56 = (vec2(2.0) + vec2(1.0));
    ivec2 unnamed_57 = (ivec2(2) - ivec2(1));
    ivec2 unnamed_58 = (ivec2(2) - ivec2(1));
    uvec2 unnamed_59 = (uvec2(2u) - uvec2(1u));
    uvec2 unnamed_60 = (uvec2(2u) - uvec2(1u));
    vec2 unnamed_61 = (vec2(2.0) - vec2(1.0));
    vec2 unnamed_62 = (vec2(2.0) - vec2(1.0));
    ivec2 unnamed_63 = (ivec2(2) * 1);
    ivec2 unnamed_64 = (2 * ivec2(1));
    uvec2 unnamed_65 = (uvec2(2u) * 1u);
    uvec2 unnamed_66 = (2u * uvec2(1u));
    vec2 unnamed_67 = (vec2(2.0) * 1.0);
    vec2 unnamed_68 = (2.0 * vec2(1.0));
    ivec2 unnamed_69 = (ivec2(2) / ivec2(1));
    ivec2 unnamed_70 = (ivec2(2) / ivec2(1));
    uvec2 unnamed_71 = (uvec2(2u) / uvec2(1u));
    uvec2 unnamed_72 = (uvec2(2u) / uvec2(1u));
    vec2 unnamed_73 = (vec2(2.0) / vec2(1.0));
    vec2 unnamed_74 = (vec2(2.0) / vec2(1.0));
    ivec2 unnamed_75 = (ivec2(2) % ivec2(1));
    ivec2 unnamed_76 = (ivec2(2) % ivec2(1));
    uvec2 unnamed_77 = (uvec2(2u) % uvec2(1u));
    uvec2 unnamed_78 = (uvec2(2u) % uvec2(1u));
    vec2 unnamed_79 = (vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0)));
    vec2 unnamed_80 = (vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0)));
    mat3x3 unnamed_81 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) + mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_82 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) - mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_83 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * 1.0);
    mat3x3 unnamed_84 = (2.0 * mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    vec3 unnamed_85 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * vec4(1.0));
    vec4 unnamed_86 = (vec3(2.0) * mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_87 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * mat3x4(vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0)));
}

void bit() {
    int unnamed_88 = (~ 1);
    uint unnamed_89 = (~ 1u);
    ivec2 unnamed_90 = (~ ivec2(1));
    uvec3 unnamed_91 = (~ uvec3(1u));
    int unnamed_92 = (2 | 1);
    uint unnamed_93 = (2u | 1u);
    ivec2 unnamed_94 = (ivec2(2) | ivec2(1));
    uvec3 unnamed_95 = (uvec3(2u) | uvec3(1u));
    int unnamed_96 = (2 & 1);
    uint unnamed_97 = (2u & 1u);
    ivec2 unnamed_98 = (ivec2(2) & ivec2(1));
    uvec3 unnamed_99 = (uvec3(2u) & uvec3(1u));
    int unnamed_100 = (2 ^ 1);
    uint unnamed_101 = (2u ^ 1u);
    ivec2 unnamed_102 = (ivec2(2) ^ ivec2(1));
    uvec3 unnamed_103 = (uvec3(2u) ^ uvec3(1u));
    int unnamed_104 = (2 << 1u);
    uint unnamed_105 = (2u << 1u);
    ivec2 unnamed_106 = (ivec2(2) << uvec2(1u));
    uvec3 unnamed_107 = (uvec3(2u) << uvec3(1u));
    int unnamed_108 = (2 >> 1u);
    uint unnamed_109 = (2u >> 1u);
    ivec2 unnamed_110 = (ivec2(2) >> uvec2(1u));
    uvec3 unnamed_111 = (uvec3(2u) >> uvec3(1u));
}

void comparison() {
    bool unnamed_112 = (2 == 1);
    bool unnamed_113 = (2u == 1u);
    bool unnamed_114 = (2.0 == 1.0);
    bvec2 unnamed_115 = equal(ivec2(2), ivec2(1));
    bvec3 unnamed_116 = equal(uvec3(2u), uvec3(1u));
    bvec4 unnamed_117 = equal(vec4(2.0), vec4(1.0));
    bool unnamed_118 = (2 != 1);
    bool unnamed_119 = (2u != 1u);
    bool unnamed_120 = (2.0 != 1.0);
    bvec2 unnamed_121 = notEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_122 = notEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_123 = notEqual(vec4(2.0), vec4(1.0));
    bool unnamed_124 = (2 < 1);
    bool unnamed_125 = (2u < 1u);
    bool unnamed_126 = (2.0 < 1.0);
    bvec2 unnamed_127 = lessThan(ivec2(2), ivec2(1));
    bvec3 unnamed_128 = lessThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_129 = lessThan(vec4(2.0), vec4(1.0));
    bool unnamed_130 = (2 <= 1);
    bool unnamed_131 = (2u <= 1u);
    bool unnamed_132 = (2.0 <= 1.0);
    bvec2 unnamed_133 = lessThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_134 = lessThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_135 = lessThanEqual(vec4(2.0), vec4(1.0));
    bool unnamed_136 = (2 > 1);
    bool unnamed_137 = (2u > 1u);
    bool unnamed_138 = (2.0 > 1.0);
    bvec2 unnamed_139 = greaterThan(ivec2(2), ivec2(1));
    bvec3 unnamed_140 = greaterThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_141 = greaterThan(vec4(2.0), vec4(1.0));
    bool unnamed_142 = (2 >= 1);
    bool unnamed_143 = (2u >= 1u);
    bool unnamed_144 = (2.0 >= 1.0);
    bvec2 unnamed_145 = greaterThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_146 = greaterThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_147 = greaterThanEqual(vec4(2.0), vec4(1.0));
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

