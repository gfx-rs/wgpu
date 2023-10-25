const v_f32_one = vec4<f32>(1.0, 1.0, 1.0, 1.0);
const v_f32_zero = vec4<f32>(0.0, 0.0, 0.0, 0.0);
const v_f32_half = vec4<f32>(0.5, 0.5, 0.5, 0.5);
const v_i32_one = vec4<i32>(1, 1, 1, 1);

fn builtins() -> vec4<f32> {
    // select()
    let condition = true;
    let s1 = select(0, 1, condition);
    let s2 = select(v_f32_zero, v_f32_one, condition);
    let s3 = select(v_f32_one, v_f32_zero, vec4<bool>(false, false, false, false));
    // mix()
    let m1 = mix(v_f32_zero, v_f32_one, v_f32_half);
    let m2 = mix(v_f32_zero, v_f32_one, 0.1);
    // bitcast()
    let b1 = bitcast<f32>(v_i32_one.x);
    let b2 = bitcast<vec4<f32>>(v_i32_one);
    // convert
    let v_i32_zero = vec4<i32>(v_f32_zero);
    // done
    return vec4<f32>(vec4<i32>(s1) + v_i32_zero) + s2 + m1 + m2 + b1 + b2;
}

fn splat() -> vec4<f32> {
    let a = (1.0 + vec2<f32>(2.0) - 3.0) / 4.0;
    let b = vec4<i32>(5) % 2;
    return a.xyxy + vec4<f32>(b);
}

fn splat_assignment() -> vec2<f32> {
    var a = vec2<f32>(2.0);
    a += 1.0;
    a -= 3.0;
    a /= 4.0;
    return a;
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

fn logical() {
    let t = true;
    let f = false;

    // unary
    let neg0 = !t;
    let neg1 = !vec2(t);

    // binary
    let or = t || f;
    let and = t && f;
    let bitwise_or0 = t | f;
    let bitwise_or1 = vec3(t) | vec3(f);
    let bitwise_and0 = t & f;
    let bitwise_and1 = vec4(t) & vec4(f);
}

fn arithmetic() {
    let one_i = 1i;
    let one_u = 1u;
    let one_f = 1.0;
    let two_i = 2i;
    let two_u = 2u;
    let two_f = 2.0;

    // unary
    let neg0 = -one_f;
    let neg1 = -vec2(one_i);
    let neg2 = -vec2(one_f);

    // binary
    // Addition
    let add0 = two_i + one_i;
    let add1 = two_u + one_u;
    let add2 = two_f + one_f;
    let add3 = vec2(two_i) + vec2(one_i);
    let add4 = vec3(two_u) + vec3(one_u);
    let add5 = vec4(two_f) + vec4(one_f);

    // Subtraction
    let sub0 = two_i - one_i;
    let sub1 = two_u - one_u;
    let sub2 = two_f - one_f;
    let sub3 = vec2(two_i) - vec2(one_i);
    let sub4 = vec3(two_u) - vec3(one_u);
    let sub5 = vec4(two_f) - vec4(one_f);

    // Multiplication
    let mul0 = two_i * one_i;
    let mul1 = two_u * one_u;
    let mul2 = two_f * one_f;
    let mul3 = vec2(two_i) * vec2(one_i);
    let mul4 = vec3(two_u) * vec3(one_u);
    let mul5 = vec4(two_f) * vec4(one_f);

    // Division
    let div0 = two_i / one_i;
    let div1 = two_u / one_u;
    let div2 = two_f / one_f;
    let div3 = vec2(two_i) / vec2(one_i);
    let div4 = vec3(two_u) / vec3(one_u);
    let div5 = vec4(two_f) / vec4(one_f);

    // Remainder
    let rem0 = two_i % one_i;
    let rem1 = two_u % one_u;
    let rem2 = two_f % one_f;
    let rem3 = vec2(two_i) % vec2(one_i);
    let rem4 = vec3(two_u) % vec3(one_u);
    let rem5 = vec4(two_f) % vec4(one_f);

    // Binary arithmetic expressions with mixed scalar and vector operands
    {
        let add0 = vec2(two_i) + one_i;
        let add1 = two_i + vec2(one_i);
        let add2 = vec2(two_u) + one_u;
        let add3 = two_u + vec2(one_u);
        let add4 = vec2(two_f) + one_f;
        let add5 = two_f + vec2(one_f);

        let sub0 = vec2(two_i) - one_i;
        let sub1 = two_i - vec2(one_i);
        let sub2 = vec2(two_u) - one_u;
        let sub3 = two_u - vec2(one_u);
        let sub4 = vec2(two_f) - one_f;
        let sub5 = two_f - vec2(one_f);

        let mul0 = vec2(two_i) * one_i;
        let mul1 = two_i * vec2(one_i);
        let mul2 = vec2(two_u) * one_u;
        let mul3 = two_u * vec2(one_u);
        let mul4 = vec2(two_f) * one_f;
        let mul5 = two_f * vec2(one_f);

        let div0 = vec2(two_i) / one_i;
        let div1 = two_i / vec2(one_i);
        let div2 = vec2(two_u) / one_u;
        let div3 = two_u / vec2(one_u);
        let div4 = vec2(two_f) / one_f;
        let div5 = two_f / vec2(one_f);

        let rem0 = vec2(two_i) % one_i;
        let rem1 = two_i % vec2(one_i);
        let rem2 = vec2(two_u) % one_u;
        let rem3 = two_u % vec2(one_u);
        let rem4 = vec2(two_f) % one_f;
        let rem5 = two_f % vec2(one_f);
    }

    // Matrix arithmetic
    let add = mat3x3<f32>() + mat3x3<f32>();
    let sub = mat3x3<f32>() - mat3x3<f32>();

    let mul_scalar0 = mat3x3<f32>() * one_f;
    let mul_scalar1 = two_f * mat3x3<f32>();

    let mul_vector0 = mat4x3<f32>() * vec4(one_f);
    let mul_vector1 = vec3f(two_f) * mat4x3f();

    let mul = mat4x3<f32>() * mat3x4<f32>();
}

fn bit() {
    let one_i = 1i;
    let one_u = 1u;
    let two_i = 2i;
    let two_u = 2u;

    // unary
    let flip0 = ~one_i;
    let flip1 = ~one_u;
    let flip2 = ~vec2(one_i);
    let flip3 = ~vec3(one_u);

    // binary
    let or0 = two_i | one_i;
    let or1 = two_u | one_u;
    let or2 = vec2(two_i) | vec2(one_i);
    let or3 = vec3(two_u) | vec3(one_u);

    let and0 = two_i & one_i;
    let and1 = two_u & one_u;
    let and2 = vec2(two_i) & vec2(one_i);
    let and3 = vec3(two_u) & vec3(one_u);

    let xor0 = two_i ^ one_i;
    let xor1 = two_u ^ one_u;
    let xor2 = vec2(two_i) ^ vec2(one_i);
    let xor3 = vec3(two_u) ^ vec3(one_u);

    let shl0 = two_i << one_u;
    let shl1 = two_u << one_u;
    let shl2 = vec2(two_i) << vec2(one_u);
    let shl3 = vec3(two_u) << vec3(one_u);

    let shr0 = two_i >> one_u;
    let shr1 = two_u >> one_u;
    let shr2 = vec2(two_i) >> vec2(one_u);
    let shr3 = vec3(two_u) >> vec3(one_u);
}

fn comparison() {
    let one_i = 1i;
    let one_u = 1u;
    let one_f = 1.0;
    let two_i = 2i;
    let two_u = 2u;
    let two_f = 2.0;

    let eq0 = two_i == one_i;
    let eq1 = two_u == one_u;
    let eq2 = two_f == one_f;
    let eq3 = vec2(two_i) == vec2(one_i);
    let eq4 = vec3(two_u) == vec3(one_u);
    let eq5 = vec4(two_f) == vec4(one_f);

    let neq0 = two_i != one_i;
    let neq1 = two_u != one_u;
    let neq2 = two_f != one_f;
    let neq3 = vec2(two_i) != vec2(one_i);
    let neq4 = vec3(two_u) != vec3(one_u);
    let neq5 = vec4(two_f) != vec4(one_f);

    let lt0 = two_i < one_i;
    let lt1 = two_u < one_u;
    let lt2 = two_f < one_f;
    let lt3 = vec2(two_i) < vec2(one_i);
    let lt4 = vec3(two_u) < vec3(one_u);
    let lt5 = vec4(two_f) < vec4(one_f);

    let lte0 = two_i <= one_i;
    let lte1 = two_u <= one_u;
    let lte2 = two_f <= one_f;
    let lte3 = vec2(two_i) <= vec2(one_i);
    let lte4 = vec3(two_u) <= vec3(one_u);
    let lte5 = vec4(two_f) <= vec4(one_f);

    let gt0 = two_i > one_i;
    let gt1 = two_u > one_u;
    let gt2 = two_f > one_f;
    let gt3 = vec2(two_i) > vec2(one_i);
    let gt4 = vec3(two_u) > vec3(one_u);
    let gt5 = vec4(two_f) > vec4(one_f);

    let gte0 = two_i >= one_i;
    let gte1 = two_u >= one_u;
    let gte2 = two_f >= one_f;
    let gte3 = vec2(two_i) >= vec2(one_i);
    let gte4 = vec3(two_u) >= vec3(one_u);
    let gte5 = vec4(two_f) >= vec4(one_f);
}

fn assignment() {
    let zero_i = 0i;
    let one_i = 1i;
    let one_u = 1u;
    let two_u = 2u;

    var a = one_i;

    a += one_i;
    a -= one_i;
    a *= a;
    a /= a;
    a %= one_i;
    a &= zero_i;
    a |= zero_i;
    a ^= zero_i;
    a <<= two_u;
    a >>= one_u;

    a++;
    a--;

    var vec0: vec3<i32> = vec3<i32>();
    vec0[one_i]++;
    vec0[one_i]--;
}

@compute @workgroup_size(1)
fn main() {
    builtins();
    splat();
    bool_cast(v_f32_one.xyz);

    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
}

fn negation_avoids_prefix_decrement() {
    let x = 1;
    let p0 = -x;
    let p1 = - -x;
    let p2 = -(-x);
    let p3 = -(- x);
    let p4 = - - -x;
    let p5 = - - - - x;
    let p6 = - - -(- -x);
    let p7 = (- - - - -x);
}
