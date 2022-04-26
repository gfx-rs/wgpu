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
    foo = Foo(vec4(1.0), 1);
    mat2x2 mat2comp = mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    mat4x4 mat4comp = mat4x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
    uvec2 unnamed = uvec2(0u);
    mat2x2 unnamed_1 = mat2x2(vec2(0.0), vec2(0.0));
    int unnamed_2[4] = int[4](0, 1, 2, 3);
    float _e59 = foo.a.x;
    return _e59;
}

void logical() {
    bool unnamed_3 = (!true);
    bvec2 unnamed_4 = not(bvec2(true));
    bool unnamed_5 = (true || false);
    bool unnamed_6 = (true && false);
    bool unnamed_7 = (true || false);
    bvec3 unnamed_8 = bvec3(bvec3(true).x || bvec3(false).x, bvec3(true).y || bvec3(false).y, bvec3(true).z || bvec3(false).z);
    bool unnamed_9 = (true && false);
    bvec4 unnamed_10 = bvec4(bvec4(true).x && bvec4(false).x, bvec4(true).y && bvec4(false).y, bvec4(true).z && bvec4(false).z, bvec4(true).w && bvec4(false).w);
}

void arithmetic() {
    ivec2 unnamed_11 = (-ivec2(1));
    vec2 unnamed_12 = (-vec2(1.0));
    int unnamed_13 = (2 + 1);
    uint unnamed_14 = (2u + 1u);
    float unnamed_15 = (2.0 + 1.0);
    ivec2 unnamed_16 = (ivec2(2) + ivec2(1));
    uvec3 unnamed_17 = (uvec3(2u) + uvec3(1u));
    vec4 unnamed_18 = (vec4(2.0) + vec4(1.0));
    int unnamed_19 = (2 - 1);
    uint unnamed_20 = (2u - 1u);
    float unnamed_21 = (2.0 - 1.0);
    ivec2 unnamed_22 = (ivec2(2) - ivec2(1));
    uvec3 unnamed_23 = (uvec3(2u) - uvec3(1u));
    vec4 unnamed_24 = (vec4(2.0) - vec4(1.0));
    int unnamed_25 = (2 * 1);
    uint unnamed_26 = (2u * 1u);
    float unnamed_27 = (2.0 * 1.0);
    ivec2 unnamed_28 = (ivec2(2) * ivec2(1));
    uvec3 unnamed_29 = (uvec3(2u) * uvec3(1u));
    vec4 unnamed_30 = (vec4(2.0) * vec4(1.0));
    int unnamed_31 = (2 / 1);
    uint unnamed_32 = (2u / 1u);
    float unnamed_33 = (2.0 / 1.0);
    ivec2 unnamed_34 = (ivec2(2) / ivec2(1));
    uvec3 unnamed_35 = (uvec3(2u) / uvec3(1u));
    vec4 unnamed_36 = (vec4(2.0) / vec4(1.0));
    int unnamed_37 = (2 % 1);
    uint unnamed_38 = (2u % 1u);
    float unnamed_39 = (2.0 - 1.0 * trunc(2.0 / 1.0));
    ivec2 unnamed_40 = (ivec2(2) % ivec2(1));
    uvec3 unnamed_41 = (uvec3(2u) % uvec3(1u));
    vec4 unnamed_42 = (vec4(2.0) - vec4(1.0) * trunc(vec4(2.0) / vec4(1.0)));
    ivec2 unnamed_43 = (ivec2(2) + ivec2(1));
    ivec2 unnamed_44 = (ivec2(2) + ivec2(1));
    uvec2 unnamed_45 = (uvec2(2u) + uvec2(1u));
    uvec2 unnamed_46 = (uvec2(2u) + uvec2(1u));
    vec2 unnamed_47 = (vec2(2.0) + vec2(1.0));
    vec2 unnamed_48 = (vec2(2.0) + vec2(1.0));
    ivec2 unnamed_49 = (ivec2(2) - ivec2(1));
    ivec2 unnamed_50 = (ivec2(2) - ivec2(1));
    uvec2 unnamed_51 = (uvec2(2u) - uvec2(1u));
    uvec2 unnamed_52 = (uvec2(2u) - uvec2(1u));
    vec2 unnamed_53 = (vec2(2.0) - vec2(1.0));
    vec2 unnamed_54 = (vec2(2.0) - vec2(1.0));
    ivec2 unnamed_55 = (ivec2(2) * 1);
    ivec2 unnamed_56 = (2 * ivec2(1));
    uvec2 unnamed_57 = (uvec2(2u) * 1u);
    uvec2 unnamed_58 = (2u * uvec2(1u));
    vec2 unnamed_59 = (vec2(2.0) * 1.0);
    vec2 unnamed_60 = (2.0 * vec2(1.0));
    ivec2 unnamed_61 = (ivec2(2) / ivec2(1));
    ivec2 unnamed_62 = (ivec2(2) / ivec2(1));
    uvec2 unnamed_63 = (uvec2(2u) / uvec2(1u));
    uvec2 unnamed_64 = (uvec2(2u) / uvec2(1u));
    vec2 unnamed_65 = (vec2(2.0) / vec2(1.0));
    vec2 unnamed_66 = (vec2(2.0) / vec2(1.0));
    ivec2 unnamed_67 = (ivec2(2) % ivec2(1));
    ivec2 unnamed_68 = (ivec2(2) % ivec2(1));
    uvec2 unnamed_69 = (uvec2(2u) % uvec2(1u));
    uvec2 unnamed_70 = (uvec2(2u) % uvec2(1u));
    vec2 unnamed_71 = (vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0)));
    vec2 unnamed_72 = (vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0)));
    mat3x3 unnamed_73 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) + mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_74 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) - mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_75 = (mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * 1.0);
    mat3x3 unnamed_76 = (2.0 * mat3x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    vec3 unnamed_77 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * vec4(1.0));
    vec4 unnamed_78 = (vec3(2.0) * mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)));
    mat3x3 unnamed_79 = (mat4x3(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)) * mat3x4(vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0)));
}

void bit() {
    int unnamed_80 = (~1);
    uint unnamed_81 = (~1u);
    ivec2 unnamed_82 = (~ivec2(1));
    uvec3 unnamed_83 = (~uvec3(1u));
    int unnamed_84 = (2 | 1);
    uint unnamed_85 = (2u | 1u);
    ivec2 unnamed_86 = (ivec2(2) | ivec2(1));
    uvec3 unnamed_87 = (uvec3(2u) | uvec3(1u));
    int unnamed_88 = (2 & 1);
    uint unnamed_89 = (2u & 1u);
    ivec2 unnamed_90 = (ivec2(2) & ivec2(1));
    uvec3 unnamed_91 = (uvec3(2u) & uvec3(1u));
    int unnamed_92 = (2 ^ 1);
    uint unnamed_93 = (2u ^ 1u);
    ivec2 unnamed_94 = (ivec2(2) ^ ivec2(1));
    uvec3 unnamed_95 = (uvec3(2u) ^ uvec3(1u));
    int unnamed_96 = (2 << 1u);
    uint unnamed_97 = (2u << 1u);
    ivec2 unnamed_98 = (ivec2(2) << uvec2(1u));
    uvec3 unnamed_99 = (uvec3(2u) << uvec3(1u));
    int unnamed_100 = (2 >> 1u);
    uint unnamed_101 = (2u >> 1u);
    ivec2 unnamed_102 = (ivec2(2) >> uvec2(1u));
    uvec3 unnamed_103 = (uvec3(2u) >> uvec3(1u));
}

void comparison() {
    bool unnamed_104 = (2 == 1);
    bool unnamed_105 = (2u == 1u);
    bool unnamed_106 = (2.0 == 1.0);
    bvec2 unnamed_107 = equal(ivec2(2), ivec2(1));
    bvec3 unnamed_108 = equal(uvec3(2u), uvec3(1u));
    bvec4 unnamed_109 = equal(vec4(2.0), vec4(1.0));
    bool unnamed_110 = (2 != 1);
    bool unnamed_111 = (2u != 1u);
    bool unnamed_112 = (2.0 != 1.0);
    bvec2 unnamed_113 = notEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_114 = notEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_115 = notEqual(vec4(2.0), vec4(1.0));
    bool unnamed_116 = (2 < 1);
    bool unnamed_117 = (2u < 1u);
    bool unnamed_118 = (2.0 < 1.0);
    bvec2 unnamed_119 = lessThan(ivec2(2), ivec2(1));
    bvec3 unnamed_120 = lessThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_121 = lessThan(vec4(2.0), vec4(1.0));
    bool unnamed_122 = (2 <= 1);
    bool unnamed_123 = (2u <= 1u);
    bool unnamed_124 = (2.0 <= 1.0);
    bvec2 unnamed_125 = lessThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_126 = lessThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_127 = lessThanEqual(vec4(2.0), vec4(1.0));
    bool unnamed_128 = (2 > 1);
    bool unnamed_129 = (2u > 1u);
    bool unnamed_130 = (2.0 > 1.0);
    bvec2 unnamed_131 = greaterThan(ivec2(2), ivec2(1));
    bvec3 unnamed_132 = greaterThan(uvec3(2u), uvec3(1u));
    bvec4 unnamed_133 = greaterThan(vec4(2.0), vec4(1.0));
    bool unnamed_134 = (2 >= 1);
    bool unnamed_135 = (2u >= 1u);
    bool unnamed_136 = (2.0 >= 1.0);
    bvec2 unnamed_137 = greaterThanEqual(ivec2(2), ivec2(1));
    bvec3 unnamed_138 = greaterThanEqual(uvec3(2u), uvec3(1u));
    bvec4 unnamed_139 = greaterThanEqual(vec4(2.0), vec4(1.0));
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

