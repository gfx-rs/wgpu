struct Foo {
    a: vec4<f32>,
    b: i32,
}

const v_f32_one: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
const v_f32_zero: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
const v_f32_half: vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.5);
const v_i32_one: vec4<i32> = vec4<i32>(1, 1, 1, 1);
fn builtins() -> vec4<f32> {
    let s1_ = select(0, 1, true);
    let s2_ = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), true);
    let s3_ = select(vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<bool>(false, false, false, false));
    let m1_ = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.5, 0.5, 0.5, 0.5));
    let m2_ = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), 0.10000000149011612);
    let b1_ = bitcast<f32>(vec4<i32>(1, 1, 1, 1).x);
    let b2_ = bitcast<vec4<f32>>(vec4<i32>(1, 1, 1, 1));
    let v_i32_zero = vec4<i32>(vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return (((((vec4<f32>((vec4<i32>(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4<f32>(b1_)) + b2_);
}

fn splat() -> vec4<f32> {
    let a_2 = (((vec2<f32>(1.0) + vec2<f32>(2.0)) - vec2<f32>(3.0)) / vec2<f32>(4.0));
    let b = (vec4<i32>(5) % vec4<i32>(2));
    return (a_2.xyxy + vec4<f32>(b));
}

fn splat_assignment() -> vec2<f32> {
    var a: vec2<f32>;

    a = vec2<f32>(2.0);
    let _e4 = a;
    a = (_e4 + vec2<f32>(1.0));
    let _e8 = a;
    a = (_e8 - vec2<f32>(3.0));
    let _e12 = a;
    a = (_e12 / vec2<f32>(4.0));
    let _e15 = a;
    return _e15;
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

fn constructors() -> f32 {
    var foo: Foo;

    foo = Foo(vec4<f32>(1.0), 1);
    let mat2comp = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    let mat4comp = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    _ = vec2<u32>(0u);
    _ = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    _ = array<i32,4u>(0, 1, 2, 3);
    _ = bool(bool());
    _ = i32(i32());
    _ = u32(u32());
    _ = f32(f32());
    _ = vec2<u32>(vec2<u32>());
    _ = mat2x3<f32>(mat2x3<f32>());
    _ = bitcast<vec2<u32>>(vec2<u32>());
    _ = mat2x3<f32>(mat2x3<f32>());
    let _e71 = foo.a.x;
    return _e71;
}

fn logical() {
    _ = !(true);
    _ = !(vec2<bool>(true));
    _ = (true || false);
    _ = (true && false);
    _ = (true | false);
    _ = (vec3<bool>(true) | vec3<bool>(false));
    _ = (true & false);
    _ = (vec4<bool>(true) & vec4<bool>(false));
}

fn arithmetic() {
    _ = -(vec2<i32>(1));
    _ = -(vec2<f32>(1.0));
    _ = (2 + 1);
    _ = (2u + 1u);
    _ = (2.0 + 1.0);
    _ = (vec2<i32>(2) + vec2<i32>(1));
    _ = (vec3<u32>(2u) + vec3<u32>(1u));
    _ = (vec4<f32>(2.0) + vec4<f32>(1.0));
    _ = (2 - 1);
    _ = (2u - 1u);
    _ = (2.0 - 1.0);
    _ = (vec2<i32>(2) - vec2<i32>(1));
    _ = (vec3<u32>(2u) - vec3<u32>(1u));
    _ = (vec4<f32>(2.0) - vec4<f32>(1.0));
    _ = (2 * 1);
    _ = (2u * 1u);
    _ = (2.0 * 1.0);
    _ = (vec2<i32>(2) * vec2<i32>(1));
    _ = (vec3<u32>(2u) * vec3<u32>(1u));
    _ = (vec4<f32>(2.0) * vec4<f32>(1.0));
    _ = (2 / 1);
    _ = (2u / 1u);
    _ = (2.0 / 1.0);
    _ = (vec2<i32>(2) / vec2<i32>(1));
    _ = (vec3<u32>(2u) / vec3<u32>(1u));
    _ = (vec4<f32>(2.0) / vec4<f32>(1.0));
    _ = (2 % 1);
    _ = (2u % 1u);
    _ = (2.0 % 1.0);
    _ = (vec2<i32>(2) % vec2<i32>(1));
    _ = (vec3<u32>(2u) % vec3<u32>(1u));
    _ = (vec4<f32>(2.0) % vec4<f32>(1.0));
    _ = (vec2<i32>(2) + vec2<i32>(1));
    _ = (vec2<i32>(2) + vec2<i32>(1));
    _ = (vec2<u32>(2u) + vec2<u32>(1u));
    _ = (vec2<u32>(2u) + vec2<u32>(1u));
    _ = (vec2<f32>(2.0) + vec2<f32>(1.0));
    _ = (vec2<f32>(2.0) + vec2<f32>(1.0));
    _ = (vec2<i32>(2) - vec2<i32>(1));
    _ = (vec2<i32>(2) - vec2<i32>(1));
    _ = (vec2<u32>(2u) - vec2<u32>(1u));
    _ = (vec2<u32>(2u) - vec2<u32>(1u));
    _ = (vec2<f32>(2.0) - vec2<f32>(1.0));
    _ = (vec2<f32>(2.0) - vec2<f32>(1.0));
    _ = (vec2<i32>(2) * 1);
    _ = (2 * vec2<i32>(1));
    _ = (vec2<u32>(2u) * 1u);
    _ = (2u * vec2<u32>(1u));
    _ = (vec2<f32>(2.0) * 1.0);
    _ = (2.0 * vec2<f32>(1.0));
    _ = (vec2<i32>(2) / vec2<i32>(1));
    _ = (vec2<i32>(2) / vec2<i32>(1));
    _ = (vec2<u32>(2u) / vec2<u32>(1u));
    _ = (vec2<u32>(2u) / vec2<u32>(1u));
    _ = (vec2<f32>(2.0) / vec2<f32>(1.0));
    _ = (vec2<f32>(2.0) / vec2<f32>(1.0));
    _ = (vec2<i32>(2) % vec2<i32>(1));
    _ = (vec2<i32>(2) % vec2<i32>(1));
    _ = (vec2<u32>(2u) % vec2<u32>(1u));
    _ = (vec2<u32>(2u) % vec2<u32>(1u));
    _ = (vec2<f32>(2.0) % vec2<f32>(1.0));
    _ = (vec2<f32>(2.0) % vec2<f32>(1.0));
    _ = (mat3x3<f32>() + mat3x3<f32>());
    _ = (mat3x3<f32>() - mat3x3<f32>());
    _ = (mat3x3<f32>() * 1.0);
    _ = (2.0 * mat3x3<f32>());
    _ = (mat4x3<f32>() * vec4<f32>(1.0));
    _ = (vec3<f32>(2.0) * mat4x3<f32>());
    _ = (mat4x3<f32>() * mat3x4<f32>());
}

fn bit() {
    _ = ~(1);
    _ = ~(1u);
    _ = !(vec2<i32>(1));
    _ = !(vec3<u32>(1u));
    _ = (2 | 1);
    _ = (2u | 1u);
    _ = (vec2<i32>(2) | vec2<i32>(1));
    _ = (vec3<u32>(2u) | vec3<u32>(1u));
    _ = (2 & 1);
    _ = (2u & 1u);
    _ = (vec2<i32>(2) & vec2<i32>(1));
    _ = (vec3<u32>(2u) & vec3<u32>(1u));
    _ = (2 ^ 1);
    _ = (2u ^ 1u);
    _ = (vec2<i32>(2) ^ vec2<i32>(1));
    _ = (vec3<u32>(2u) ^ vec3<u32>(1u));
    _ = (2 << 1u);
    _ = (2u << 1u);
    _ = (vec2<i32>(2) << vec2<u32>(1u));
    _ = (vec3<u32>(2u) << vec3<u32>(1u));
    _ = (2 >> 1u);
    _ = (2u >> 1u);
    _ = (vec2<i32>(2) >> vec2<u32>(1u));
    _ = (vec3<u32>(2u) >> vec3<u32>(1u));
}

fn comparison() {
    _ = (2 == 1);
    _ = (2u == 1u);
    _ = (2.0 == 1.0);
    _ = (vec2<i32>(2) == vec2<i32>(1));
    _ = (vec3<u32>(2u) == vec3<u32>(1u));
    _ = (vec4<f32>(2.0) == vec4<f32>(1.0));
    _ = (2 != 1);
    _ = (2u != 1u);
    _ = (2.0 != 1.0);
    _ = (vec2<i32>(2) != vec2<i32>(1));
    _ = (vec3<u32>(2u) != vec3<u32>(1u));
    _ = (vec4<f32>(2.0) != vec4<f32>(1.0));
    _ = (2 < 1);
    _ = (2u < 1u);
    _ = (2.0 < 1.0);
    _ = (vec2<i32>(2) < vec2<i32>(1));
    _ = (vec3<u32>(2u) < vec3<u32>(1u));
    _ = (vec4<f32>(2.0) < vec4<f32>(1.0));
    _ = (2 <= 1);
    _ = (2u <= 1u);
    _ = (2.0 <= 1.0);
    _ = (vec2<i32>(2) <= vec2<i32>(1));
    _ = (vec3<u32>(2u) <= vec3<u32>(1u));
    _ = (vec4<f32>(2.0) <= vec4<f32>(1.0));
    _ = (2 > 1);
    _ = (2u > 1u);
    _ = (2.0 > 1.0);
    _ = (vec2<i32>(2) > vec2<i32>(1));
    _ = (vec3<u32>(2u) > vec3<u32>(1u));
    _ = (vec4<f32>(2.0) > vec4<f32>(1.0));
    _ = (2 >= 1);
    _ = (2u >= 1u);
    _ = (2.0 >= 1.0);
    _ = (vec2<i32>(2) >= vec2<i32>(1));
    _ = (vec3<u32>(2u) >= vec3<u32>(1u));
    _ = (vec4<f32>(2.0) >= vec4<f32>(1.0));
}

fn assignment() {
    var a_1: i32;
    var vec0_: vec3<i32>;

    a_1 = 1;
    let _e3 = a_1;
    a_1 = (_e3 + 1);
    let _e6 = a_1;
    a_1 = (_e6 - 1);
    let _e8 = a_1;
    let _e9 = a_1;
    a_1 = (_e9 * _e8);
    let _e11 = a_1;
    let _e12 = a_1;
    a_1 = (_e12 / _e11);
    let _e15 = a_1;
    a_1 = (_e15 % 1);
    let _e18 = a_1;
    a_1 = (_e18 & 0);
    let _e21 = a_1;
    a_1 = (_e21 | 0);
    let _e24 = a_1;
    a_1 = (_e24 ^ 0);
    let _e27 = a_1;
    a_1 = (_e27 << 2u);
    let _e30 = a_1;
    a_1 = (_e30 >> 1u);
    let _e32 = a_1;
    a_1 = (_e32 + 1);
    let _e35 = a_1;
    a_1 = (_e35 - 1);
    vec0_ = vec3<i32>();
    let _e42 = vec0_.y;
    vec0_.y = (_e42 + 1);
    let _e47 = vec0_.y;
    vec0_.y = (_e47 - 1);
    return;
}

fn negation_avoids_prefix_decrement() {
    _ = -(-2);
    _ = -(-3);
    _ = -(-(4));
    _ = -(-(-5));
    _ = -(-(-(-(6))));
    _ = -(-(-(-(-7))));
    _ = -(-(-(-(-8))));
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = builtins();
    let _e1 = splat();
    let _e4 = bool_cast(vec4<f32>(1.0, 1.0, 1.0, 1.0).xyz);
    let _e5 = constructors();
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
