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
    // unary
    let neg0 = !true;
    let neg1 = !vec2(true);

    // binary
    let or = true || false;
    let and = true && false;
    let bitwise_or0 = true | false;
    let bitwise_or1 = vec3(true) | vec3(false);
    let bitwise_and0 = true & false;
    let bitwise_and1 = vec4(true) & vec4(false);
}

fn arithmetic() {
    // unary
    let neg0 = -1.0;
    let neg1 = -vec2(1);
    let neg2 = -vec2(1.0);

    // binary
    // Addition
    let add0 = 2 + 1;
    let add1 = 2u + 1u;
    let add2 = 2.0 + 1.0;
    let add3 = vec2(2) + vec2(1);
    let add4 = vec3(2u) + vec3(1u);
    let add5 = vec4(2.0) + vec4(1.0);

    // Subtraction
    let sub0 = 2 - 1;
    let sub1 = 2u - 1u;
    let sub2 = 2.0 - 1.0;
    let sub3 = vec2(2) - vec2(1);
    let sub4 = vec3(2u) - vec3(1u);
    let sub5 = vec4(2.0) - vec4(1.0);

    // Multiplication
    let mul0 = 2 * 1;
    let mul1 = 2u * 1u;
    let mul2 = 2.0 * 1.0;
    let mul3 = vec2(2) * vec2(1);
    let mul4 = vec3(2u) * vec3(1u);
    let mul5 = vec4(2.0) * vec4(1.0);

    // Division
    let div0 = 2 / 1;
    let div1 = 2u / 1u;
    let div2 = 2.0 / 1.0;
    let div3 = vec2(2) / vec2(1);
    let div4 = vec3(2u) / vec3(1u);
    let div5 = vec4(2.0) / vec4(1.0);

    // Remainder
    let rem0 = 2 % 1;
    let rem1 = 2u % 1u;
    let rem2 = 2.0 % 1.0;
    let rem3 = vec2(2) % vec2(1);
    let rem4 = vec3(2u) % vec3(1u);
    let rem5 = vec4(2.0) % vec4(1.0);

    // Binary arithmetic expressions with mixed scalar and vector operands
    {
        let add0 = vec2(2) + 1;
        let add1 = 2 + vec2(1);
        let add2 = vec2(2u) + 1u;
        let add3 = 2u + vec2(1u);
        let add4 = vec2(2.0) + 1.0;
        let add5 = 2.0 + vec2(1.0);

        let sub0 = vec2(2) - 1;
        let sub1 = 2 - vec2(1);
        let sub2 = vec2(2u) - 1u;
        let sub3 = 2u - vec2(1u);
        let sub4 = vec2(2.0) - 1.0;
        let sub5 = 2.0 - vec2(1.0);

        let mul0 = vec2(2) * 1;
        let mul1 = 2 * vec2(1);
        let mul2 = vec2(2u) * 1u;
        let mul3 = 2u * vec2(1u);
        let mul4 = vec2(2.0) * 1.0;
        let mul5 = 2.0 * vec2(1.0);

        let div0 = vec2(2) / 1;
        let div1 = 2 / vec2(1);
        let div2 = vec2(2u) / 1u;
        let div3 = 2u / vec2(1u);
        let div4 = vec2(2.0) / 1.0;
        let div5 = 2.0 / vec2(1.0);

        let rem0 = vec2(2) % 1;
        let rem1 = 2 % vec2(1);
        let rem2 = vec2(2u) % 1u;
        let rem3 = 2u % vec2(1u);
        let rem4 = vec2(2.0) % 1.0;
        let rem5 = 2.0 % vec2(1.0);
    }

    // Matrix arithmetic
    let add = mat3x3<f32>() + mat3x3<f32>();
    let sub = mat3x3<f32>() - mat3x3<f32>();

    let mul_scalar0 = mat3x3<f32>() * 1.0;
    let mul_scalar1 = 2.0 * mat3x3<f32>();

    let mul_vector0 = mat4x3<f32>() * vec4(1.0);
    let mul_vector1 = vec3f(2.0) * mat4x3f();

    let mul = mat4x3<f32>() * mat3x4<f32>();
}

fn bit() {
    // unary
    let flip0 = ~1;
    let flip1 = ~1u;
    let flip2 = ~vec2(1);
    let flip3 = ~vec3(1u);

    // binary
    let or0 = 2 | 1;
    let or1 = 2u | 1u;
    let or2 = vec2(2) | vec2(1);
    let or3 = vec3(2u) | vec3(1u);

    let and0 = 2 & 1;
    let and1 = 2u & 1u;
    let and2 = vec2(2) & vec2(1);
    let and3 = vec3(2u) & vec3(1u);

    let xor0 = 2 ^ 1;
    let xor1 = 2u ^ 1u;
    let xor2 = vec2(2) ^ vec2(1);
    let xor3 = vec3(2u) ^ vec3(1u);

    let shl0 = 2 << 1u;
    let shl1 = 2u << 1u;
    let shl2 = vec2(2) << vec2(1u);
    let shl3 = vec3(2u) << vec3(1u);

    let shr0 = 2 >> 1u;
    let shr1 = 2u >> 1u;
    let shr2 = vec2(2) >> vec2(1u);
    let shr3 = vec3(2u) >> vec3(1u);
}

fn comparison() {
    let eq0 = 2 == 1;
    let eq1 = 2u == 1u;
    let eq2 = 2.0 == 1.0;
    let eq3 = vec2(2) == vec2(1);
    let eq4 = vec3(2u) == vec3(1u);
    let eq5 = vec4(2.0) == vec4(1.0);

    let neq0 = 2 != 1;
    let neq1 = 2u != 1u;
    let neq2 = 2.0 != 1.0;
    let neq3 = vec2(2) != vec2(1);
    let neq4 = vec3(2u) != vec3(1u);
    let neq5 = vec4(2.0) != vec4(1.0);

    let lt0 = 2 < 1;
    let lt1 = 2u < 1u;
    let lt2 = 2.0 < 1.0;
    let lt3 = vec2(2) < vec2(1);
    let lt4 = vec3(2u) < vec3(1u);
    let lt5 = vec4(2.0) < vec4(1.0);

    let lte0 = 2 <= 1;
    let lte1 = 2u <= 1u;
    let lte2 = 2.0 <= 1.0;
    let lte3 = vec2(2) <= vec2(1);
    let lte4 = vec3(2u) <= vec3(1u);
    let lte5 = vec4(2.0) <= vec4(1.0);

    let gt0 = 2 > 1;
    let gt1 = 2u > 1u;
    let gt2 = 2.0 > 1.0;
    let gt3 = vec2(2) > vec2(1);
    let gt4 = vec3(2u) > vec3(1u);
    let gt5 = vec4(2.0) > vec4(1.0);

    let gte0 = 2 >= 1;
    let gte1 = 2u >= 1u;
    let gte2 = 2.0 >= 1.0;
    let gte3 = vec2(2) >= vec2(1);
    let gte4 = vec3(2u) >= vec3(1u);
    let gte5 = vec4(2.0) >= vec4(1.0);
}

fn assignment() {
    var a = 1;

    a += 1;
    a -= 1;
    a *= a;
    a /= a;
    a %= 1;
    a &= 0;
    a |= 0;
    a ^= 0;
    a <<= 2u;
    a >>= 1u;

    a++;
    a--;

    var vec0: vec3<i32> = vec3<i32>();
    vec0[1]++;
    vec0[1]--;
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
    let p0 = -1;
    let p1 = - -2;
    let p2 = -(-3);
    let p3 = -(- 4);
    let p4 = - - -5;
    let p5 = - - - - 6;
    let p6 = - - -(- -7);
    let p7 = (- - - - -8);
}
