#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Foo {
    vec4 a;
    int b;
};
const vec4 v_f32_one = vec4(1.0, 1.0, 1.0, 1.0);
const vec4 v_f32_zero = vec4(0.0, 0.0, 0.0, 0.0);
const vec4 v_f32_half = vec4(0.5, 0.5, 0.5, 0.5);
const ivec4 v_i32_one = ivec4(1, 1, 1, 1);


vec4 builtins() {
    int s1_ = (true ? 1 : 0);
    vec4 s2_ = (true ? v_f32_one : v_f32_zero);
    vec4 s3_ = mix(v_f32_one, v_f32_zero, bvec4(false, false, false, false));
    vec4 m1_ = mix(v_f32_zero, v_f32_one, v_f32_half);
    vec4 m2_ = mix(v_f32_zero, v_f32_one, 0.1);
    float b1_ = intBitsToFloat(v_i32_one.x);
    vec4 b2_ = intBitsToFloat(v_i32_one);
    ivec4 v_i32_zero = ivec4(v_f32_zero);
    return (((((vec4((ivec4(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4(b1_)) + b2_);
}

vec4 splat() {
    vec2 a_2 = (((vec2(1.0) + vec2(2.0)) - vec2(3.0)) / vec2(4.0));
    ivec4 b = (ivec4(5) % ivec4(2));
    return (a_2.xyxy + vec4(b));
}

vec2 splat_assignment() {
    vec2 a = vec2(0.0);
    a = vec2(2.0);
    vec2 _e4 = a;
    a = (_e4 + vec2(1.0));
    vec2 _e8 = a;
    a = (_e8 - vec2(3.0));
    vec2 _e12 = a;
    a = (_e12 / vec2(4.0));
    vec2 _e15 = a;
    return _e15;
}

vec3 bool_cast(vec3 x) {
    bvec3 y = bvec3(x);
    return vec3(y);
}

float constructors() {
    Foo foo = Foo(vec4(0.0), 0);
    foo = Foo(vec4(1.0), 1);
    mat2x2 m0_ = mat2x2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    mat4x4 m1_1 = mat4x4(vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0));
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
    float _e71 = foo.a.x;
    return _e71;
}

void logical() {
    bool neg0_ = !(true);
    bvec2 neg1_ = not(bvec2(true));
    bool or = (true || false);
    bool and = (true && false);
    bool bitwise_or0_ = (true || false);
    bvec3 bitwise_or1_ = bvec3(bvec3(true).x || bvec3(false).x, bvec3(true).y || bvec3(false).y, bvec3(true).z || bvec3(false).z);
    bool bitwise_and0_ = (true && false);
    bvec4 bitwise_and1_ = bvec4(bvec4(true).x && bvec4(false).x, bvec4(true).y && bvec4(false).y, bvec4(true).z && bvec4(false).z, bvec4(true).w && bvec4(false).w);
}

void arithmetic() {
    ivec2 neg1_1 = -(ivec2(1));
    vec2 neg2_ = -(vec2(1.0));
    int add0_ = (2 + 1);
    uint add1_ = (2u + 1u);
    float add2_ = (2.0 + 1.0);
    ivec2 add3_ = (ivec2(2) + ivec2(1));
    uvec3 add4_ = (uvec3(2u) + uvec3(1u));
    vec4 add5_ = (vec4(2.0) + vec4(1.0));
    int sub0_ = (2 - 1);
    uint sub1_ = (2u - 1u);
    float sub2_ = (2.0 - 1.0);
    ivec2 sub3_ = (ivec2(2) - ivec2(1));
    uvec3 sub4_ = (uvec3(2u) - uvec3(1u));
    vec4 sub5_ = (vec4(2.0) - vec4(1.0));
    int mul0_ = (2 * 1);
    uint mul1_ = (2u * 1u);
    float mul2_ = (2.0 * 1.0);
    ivec2 mul3_ = (ivec2(2) * ivec2(1));
    uvec3 mul4_ = (uvec3(2u) * uvec3(1u));
    vec4 mul5_ = (vec4(2.0) * vec4(1.0));
    int div0_ = (2 / 1);
    uint div1_ = (2u / 1u);
    float div2_ = (2.0 / 1.0);
    ivec2 div3_ = (ivec2(2) / ivec2(1));
    uvec3 div4_ = (uvec3(2u) / uvec3(1u));
    vec4 div5_ = (vec4(2.0) / vec4(1.0));
    int rem0_ = (2 % 1);
    uint rem1_ = (2u % 1u);
    float rem2_ = (2.0 - 1.0 * trunc(2.0 / 1.0));
    ivec2 rem3_ = (ivec2(2) % ivec2(1));
    uvec3 rem4_ = (uvec3(2u) % uvec3(1u));
    vec4 rem5_ = (vec4(2.0) - vec4(1.0) * trunc(vec4(2.0) / vec4(1.0)));
    {
        ivec2 add0_1 = (ivec2(2) + ivec2(1));
        ivec2 add1_1 = (ivec2(2) + ivec2(1));
        uvec2 add2_1 = (uvec2(2u) + uvec2(1u));
        uvec2 add3_1 = (uvec2(2u) + uvec2(1u));
        vec2 add4_1 = (vec2(2.0) + vec2(1.0));
        vec2 add5_1 = (vec2(2.0) + vec2(1.0));
        ivec2 sub0_1 = (ivec2(2) - ivec2(1));
        ivec2 sub1_1 = (ivec2(2) - ivec2(1));
        uvec2 sub2_1 = (uvec2(2u) - uvec2(1u));
        uvec2 sub3_1 = (uvec2(2u) - uvec2(1u));
        vec2 sub4_1 = (vec2(2.0) - vec2(1.0));
        vec2 sub5_1 = (vec2(2.0) - vec2(1.0));
        ivec2 mul0_1 = (ivec2(2) * 1);
        ivec2 mul1_1 = (2 * ivec2(1));
        uvec2 mul2_1 = (uvec2(2u) * 1u);
        uvec2 mul3_1 = (2u * uvec2(1u));
        vec2 mul4_1 = (vec2(2.0) * 1.0);
        vec2 mul5_1 = (2.0 * vec2(1.0));
        ivec2 div0_1 = (ivec2(2) / ivec2(1));
        ivec2 div1_1 = (ivec2(2) / ivec2(1));
        uvec2 div2_1 = (uvec2(2u) / uvec2(1u));
        uvec2 div3_1 = (uvec2(2u) / uvec2(1u));
        vec2 div4_1 = (vec2(2.0) / vec2(1.0));
        vec2 div5_1 = (vec2(2.0) / vec2(1.0));
        ivec2 rem0_1 = (ivec2(2) % ivec2(1));
        ivec2 rem1_1 = (ivec2(2) % ivec2(1));
        uvec2 rem2_1 = (uvec2(2u) % uvec2(1u));
        uvec2 rem3_1 = (uvec2(2u) % uvec2(1u));
        vec2 rem4_1 = (vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0)));
        vec2 rem5_1 = (vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0)));
    }
    mat3x3 add = (mat3x3(0.0) + mat3x3(0.0));
    mat3x3 sub = (mat3x3(0.0) - mat3x3(0.0));
    mat3x3 mul_scalar0_ = (mat3x3(0.0) * 1.0);
    mat3x3 mul_scalar1_ = (2.0 * mat3x3(0.0));
    vec3 mul_vector0_ = (mat4x3(0.0) * vec4(1.0));
    vec4 mul_vector1_ = (vec3(2.0) * mat4x3(0.0));
    mat3x3 mul = (mat4x3(0.0) * mat3x4(0.0));
}

void bit() {
    int flip0_ = ~(1);
    uint flip1_ = ~(1u);
    ivec2 flip2_ = ~(ivec2(1));
    uvec3 flip3_ = ~(uvec3(1u));
    int or0_ = (2 | 1);
    uint or1_ = (2u | 1u);
    ivec2 or2_ = (ivec2(2) | ivec2(1));
    uvec3 or3_ = (uvec3(2u) | uvec3(1u));
    int and0_ = (2 & 1);
    uint and1_ = (2u & 1u);
    ivec2 and2_ = (ivec2(2) & ivec2(1));
    uvec3 and3_ = (uvec3(2u) & uvec3(1u));
    int xor0_ = (2 ^ 1);
    uint xor1_ = (2u ^ 1u);
    ivec2 xor2_ = (ivec2(2) ^ ivec2(1));
    uvec3 xor3_ = (uvec3(2u) ^ uvec3(1u));
    int shl0_ = (2 << 1u);
    uint shl1_ = (2u << 1u);
    ivec2 shl2_ = (ivec2(2) << uvec2(1u));
    uvec3 shl3_ = (uvec3(2u) << uvec3(1u));
    int shr0_ = (2 >> 1u);
    uint shr1_ = (2u >> 1u);
    ivec2 shr2_ = (ivec2(2) >> uvec2(1u));
    uvec3 shr3_ = (uvec3(2u) >> uvec3(1u));
}

void comparison() {
    bool eq0_ = (2 == 1);
    bool eq1_ = (2u == 1u);
    bool eq2_ = (2.0 == 1.0);
    bvec2 eq3_ = equal(ivec2(2), ivec2(1));
    bvec3 eq4_ = equal(uvec3(2u), uvec3(1u));
    bvec4 eq5_ = equal(vec4(2.0), vec4(1.0));
    bool neq0_ = (2 != 1);
    bool neq1_ = (2u != 1u);
    bool neq2_ = (2.0 != 1.0);
    bvec2 neq3_ = notEqual(ivec2(2), ivec2(1));
    bvec3 neq4_ = notEqual(uvec3(2u), uvec3(1u));
    bvec4 neq5_ = notEqual(vec4(2.0), vec4(1.0));
    bool lt0_ = (2 < 1);
    bool lt1_ = (2u < 1u);
    bool lt2_ = (2.0 < 1.0);
    bvec2 lt3_ = lessThan(ivec2(2), ivec2(1));
    bvec3 lt4_ = lessThan(uvec3(2u), uvec3(1u));
    bvec4 lt5_ = lessThan(vec4(2.0), vec4(1.0));
    bool lte0_ = (2 <= 1);
    bool lte1_ = (2u <= 1u);
    bool lte2_ = (2.0 <= 1.0);
    bvec2 lte3_ = lessThanEqual(ivec2(2), ivec2(1));
    bvec3 lte4_ = lessThanEqual(uvec3(2u), uvec3(1u));
    bvec4 lte5_ = lessThanEqual(vec4(2.0), vec4(1.0));
    bool gt0_ = (2 > 1);
    bool gt1_ = (2u > 1u);
    bool gt2_ = (2.0 > 1.0);
    bvec2 gt3_ = greaterThan(ivec2(2), ivec2(1));
    bvec3 gt4_ = greaterThan(uvec3(2u), uvec3(1u));
    bvec4 gt5_ = greaterThan(vec4(2.0), vec4(1.0));
    bool gte0_ = (2 >= 1);
    bool gte1_ = (2u >= 1u);
    bool gte2_ = (2.0 >= 1.0);
    bvec2 gte3_ = greaterThanEqual(ivec2(2), ivec2(1));
    bvec3 gte4_ = greaterThanEqual(uvec3(2u), uvec3(1u));
    bvec4 gte5_ = greaterThanEqual(vec4(2.0), vec4(1.0));
}

void assignment() {
    int a_1 = 0;
    ivec3 vec0_ = ivec3(0);
    a_1 = 1;
    int _e3 = a_1;
    a_1 = (_e3 + 1);
    int _e6 = a_1;
    a_1 = (_e6 - 1);
    int _e8 = a_1;
    int _e9 = a_1;
    a_1 = (_e9 * _e8);
    int _e11 = a_1;
    int _e12 = a_1;
    a_1 = (_e12 / _e11);
    int _e15 = a_1;
    a_1 = (_e15 % 1);
    int _e18 = a_1;
    a_1 = (_e18 & 0);
    int _e21 = a_1;
    a_1 = (_e21 | 0);
    int _e24 = a_1;
    a_1 = (_e24 ^ 0);
    int _e27 = a_1;
    a_1 = (_e27 << 2u);
    int _e30 = a_1;
    a_1 = (_e30 >> 1u);
    int _e33 = a_1;
    a_1 = (_e33 + 1);
    int _e36 = a_1;
    a_1 = (_e36 - 1);
    vec0_ = ivec3(0);
    int _e43 = vec0_.y;
    vec0_.y = (_e43 + 1);
    int _e48 = vec0_.y;
    vec0_.y = (_e48 - 1);
    return;
}

void negation_avoids_prefix_decrement() {
    int p1_ = -(-2);
    int p2_ = -(-3);
    int p3_ = -(-(4));
    int p4_ = -(-(-5));
    int p5_ = -(-(-(-(6))));
    int p6_ = -(-(-(-(-7))));
    int p7_ = -(-(-(-(-8))));
}

void main() {
    vec4 _e0 = builtins();
    vec4 _e1 = splat();
    vec3 _e4 = bool_cast(v_f32_one.xyz);
    float _e5 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}

