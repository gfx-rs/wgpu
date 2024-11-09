const v_f32_one: vec4<f32> = vec4<f32>(1f, 1f, 1f, 1f);
const v_f32_zero: vec4<f32> = vec4<f32>(0f, 0f, 0f, 0f);
const v_f32_half: vec4<f32> = vec4<f32>(0.5f, 0.5f, 0.5f, 0.5f);
const v_i32_one: vec4<i32> = vec4<i32>(1i, 1i, 1i, 1i);

fn builtins() -> vec4<f32> {
    let s1_ = select(0i, 1i, true);
    let s2_ = select(v_f32_zero, v_f32_one, true);
    let s3_ = select(v_f32_one, v_f32_zero, vec4<bool>(false, false, false, false));
    let m1_ = mix(v_f32_zero, v_f32_one, v_f32_half);
    let m2_ = mix(v_f32_zero, v_f32_one, 0.1f);
    let b1_ = bitcast<f32>(1i);
    let b2_ = bitcast<vec4<f32>>(v_i32_one);
    const v_i32_zero = vec4<i32>(0i, 0i, 0i, 0i);
    return (((((vec4<f32>((vec4(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4(b1_)) + b2_);
}

fn splat(m: f32, n: i32) -> vec4<f32> {
    let a_2 = (((vec2(2f) + vec2(m)) - vec2(4f)) / vec2(8f));
    let b = (vec4(n) % vec4(2i));
    return (a_2.xyxy + vec4<f32>(b));
}

fn splat_assignment() -> vec2<f32> {
    var a: vec2<f32> = vec2(2f);

    let _e4 = a;
    a = (_e4 + vec2(1f));
    let _e8 = a;
    a = (_e8 - vec2(3f));
    let _e12 = a;
    a = (_e12 / vec2(4f));
    let _e15 = a;
    return _e15;
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

fn logical() {
    const neg0_ = !(true);
    const neg1_ = !(vec2(true));
    let or = (true || false);
    let and = (true && false);
    let bitwise_or0_ = (true | false);
    let bitwise_or1_ = (vec3(true) | vec3(false));
    let bitwise_and0_ = (true & false);
    let bitwise_and1_ = (vec4(true) & vec4(false));
}

fn arithmetic() {
    const neg0_1 = -(1f);
    const neg1_1 = -(vec2(1i));
    const neg2_ = -(vec2(1f));
    let add0_ = (2i + 1i);
    let add1_ = (2u + 1u);
    let add2_ = (2f + 1f);
    let add3_ = (vec2(2i) + vec2(1i));
    let add4_ = (vec3(2u) + vec3(1u));
    let add5_ = (vec4(2f) + vec4(1f));
    let sub0_ = (2i - 1i);
    let sub1_ = (2u - 1u);
    let sub2_ = (2f - 1f);
    let sub3_ = (vec2(2i) - vec2(1i));
    let sub4_ = (vec3(2u) - vec3(1u));
    let sub5_ = (vec4(2f) - vec4(1f));
    let mul0_ = (2i * 1i);
    let mul1_ = (2u * 1u);
    let mul2_ = (2f * 1f);
    let mul3_ = (vec2(2i) * vec2(1i));
    let mul4_ = (vec3(2u) * vec3(1u));
    let mul5_ = (vec4(2f) * vec4(1f));
    let div0_ = (2i / 1i);
    let div1_ = (2u / 1u);
    let div2_ = (2f / 1f);
    let div3_ = (vec2(2i) / vec2(1i));
    let div4_ = (vec3(2u) / vec3(1u));
    let div5_ = (vec4(2f) / vec4(1f));
    let rem0_ = (2i % 1i);
    let rem1_ = (2u % 1u);
    let rem2_ = (2f % 1f);
    let rem3_ = (vec2(2i) % vec2(1i));
    let rem4_ = (vec3(2u) % vec3(1u));
    let rem5_ = (vec4(2f) % vec4(1f));
    {
        let add0_1 = (vec2(2i) + vec2(1i));
        let add1_1 = (vec2(2i) + vec2(1i));
        let add2_1 = (vec2(2u) + vec2(1u));
        let add3_1 = (vec2(2u) + vec2(1u));
        let add4_1 = (vec2(2f) + vec2(1f));
        let add5_1 = (vec2(2f) + vec2(1f));
        let sub0_1 = (vec2(2i) - vec2(1i));
        let sub1_1 = (vec2(2i) - vec2(1i));
        let sub2_1 = (vec2(2u) - vec2(1u));
        let sub3_1 = (vec2(2u) - vec2(1u));
        let sub4_1 = (vec2(2f) - vec2(1f));
        let sub5_1 = (vec2(2f) - vec2(1f));
        let mul0_1 = (vec2(2i) * 1i);
        let mul1_1 = (2i * vec2(1i));
        let mul2_1 = (vec2(2u) * 1u);
        let mul3_1 = (2u * vec2(1u));
        let mul4_1 = (vec2(2f) * 1f);
        let mul5_1 = (2f * vec2(1f));
        let div0_1 = (vec2(2i) / vec2(1i));
        let div1_1 = (vec2(2i) / vec2(1i));
        let div2_1 = (vec2(2u) / vec2(1u));
        let div3_1 = (vec2(2u) / vec2(1u));
        let div4_1 = (vec2(2f) / vec2(1f));
        let div5_1 = (vec2(2f) / vec2(1f));
        let rem0_1 = (vec2(2i) % vec2(1i));
        let rem1_1 = (vec2(2i) % vec2(1i));
        let rem2_1 = (vec2(2u) % vec2(1u));
        let rem3_1 = (vec2(2u) % vec2(1u));
        let rem4_1 = (vec2(2f) % vec2(1f));
        let rem5_1 = (vec2(2f) % vec2(1f));
    }
    let add = (mat3x3<f32>() + mat3x3<f32>());
    let sub = (mat3x3<f32>() - mat3x3<f32>());
    let mul_scalar0_ = (mat3x3<f32>() * 1f);
    let mul_scalar1_ = (2f * mat3x3<f32>());
    let mul_vector0_ = (mat4x3<f32>() * vec4(1f));
    let mul_vector1_ = (vec3(2f) * mat4x3<f32>());
    let mul = (mat4x3<f32>() * mat3x4<f32>());
}

fn bit() {
    const flip0_ = ~(1i);
    const flip1_ = ~(1u);
    const flip2_ = ~(vec2(1i));
    const flip3_ = ~(vec3(1u));
    let or0_ = (2i | 1i);
    let or1_ = (2u | 1u);
    let or2_ = (vec2(2i) | vec2(1i));
    let or3_ = (vec3(2u) | vec3(1u));
    let and0_ = (2i & 1i);
    let and1_ = (2u & 1u);
    let and2_ = (vec2(2i) & vec2(1i));
    let and3_ = (vec3(2u) & vec3(1u));
    let xor0_ = (2i ^ 1i);
    let xor1_ = (2u ^ 1u);
    let xor2_ = (vec2(2i) ^ vec2(1i));
    let xor3_ = (vec3(2u) ^ vec3(1u));
    let shl0_ = (2i << 1u);
    let shl1_ = (2u << 1u);
    let shl2_ = (vec2(2i) << vec2(1u));
    let shl3_ = (vec3(2u) << vec3(1u));
    let shr0_ = (2i >> 1u);
    let shr1_ = (2u >> 1u);
    let shr2_ = (vec2(2i) >> vec2(1u));
    let shr3_ = (vec3(2u) >> vec3(1u));
}

fn comparison() {
    let eq0_ = (2i == 1i);
    let eq1_ = (2u == 1u);
    let eq2_ = (2f == 1f);
    let eq3_ = (vec2(2i) == vec2(1i));
    let eq4_ = (vec3(2u) == vec3(1u));
    let eq5_ = (vec4(2f) == vec4(1f));
    let neq0_ = (2i != 1i);
    let neq1_ = (2u != 1u);
    let neq2_ = (2f != 1f);
    let neq3_ = (vec2(2i) != vec2(1i));
    let neq4_ = (vec3(2u) != vec3(1u));
    let neq5_ = (vec4(2f) != vec4(1f));
    let lt0_ = (2i < 1i);
    let lt1_ = (2u < 1u);
    let lt2_ = (2f < 1f);
    let lt3_ = (vec2(2i) < vec2(1i));
    let lt4_ = (vec3(2u) < vec3(1u));
    let lt5_ = (vec4(2f) < vec4(1f));
    let lte0_ = (2i <= 1i);
    let lte1_ = (2u <= 1u);
    let lte2_ = (2f <= 1f);
    let lte3_ = (vec2(2i) <= vec2(1i));
    let lte4_ = (vec3(2u) <= vec3(1u));
    let lte5_ = (vec4(2f) <= vec4(1f));
    let gt0_ = (2i > 1i);
    let gt1_ = (2u > 1u);
    let gt2_ = (2f > 1f);
    let gt3_ = (vec2(2i) > vec2(1i));
    let gt4_ = (vec3(2u) > vec3(1u));
    let gt5_ = (vec4(2f) > vec4(1f));
    let gte0_ = (2i >= 1i);
    let gte1_ = (2u >= 1u);
    let gte2_ = (2f >= 1f);
    let gte3_ = (vec2(2i) >= vec2(1i));
    let gte4_ = (vec3(2u) >= vec3(1u));
    let gte5_ = (vec4(2f) >= vec4(1f));
}

fn assignment() {
    var a_1: i32;
    var vec0_: vec3<i32> = vec3<i32>();

    a_1 = 1i;
    let _e5 = a_1;
    a_1 = (_e5 + 1i);
    let _e7 = a_1;
    a_1 = (_e7 - 1i);
    let _e9 = a_1;
    let _e10 = a_1;
    a_1 = (_e10 * _e9);
    let _e12 = a_1;
    let _e13 = a_1;
    a_1 = (_e13 / _e12);
    let _e15 = a_1;
    a_1 = (_e15 % 1i);
    let _e17 = a_1;
    a_1 = (_e17 & 0i);
    let _e19 = a_1;
    a_1 = (_e19 | 0i);
    let _e21 = a_1;
    a_1 = (_e21 ^ 0i);
    let _e23 = a_1;
    a_1 = (_e23 << 2u);
    let _e25 = a_1;
    a_1 = (_e25 >> 1u);
    let _e28 = a_1;
    a_1 = (_e28 + 1i);
    let _e31 = a_1;
    a_1 = (_e31 - 1i);
    let _e37 = vec0_[1i];
    vec0_[1i] = (_e37 + 1i);
    let _e41 = vec0_[1i];
    vec0_[1i] = (_e41 - 1i);
    return;
}

fn negation_avoids_prefix_decrement() {
    const p0_ = -(1i);
    const p1_ = -(-(1i));
    const p2_ = -(-(1i));
    const p3_ = -(-(1i));
    const p4_ = -(-(-(1i)));
    const p5_ = -(-(-(-(1i))));
    const p6_ = -(-(-(-(-(1i)))));
    const p7_ = -(-(-(-(-(1i)))));
}

@compute @workgroup_size(1, 1, 1) 
fn main(@builtin(workgroup_id) id: vec3<u32>) {
    let _e1 = builtins();
    let _e6 = splat(f32(id.x), i32(id.y));
    let _e11 = bool_cast(vec3<f32>(1f, 1f, 1f));
    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
    return;
}
