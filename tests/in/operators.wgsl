//TODO: support splatting constructors for globals?
let v_f32_one = vec4<f32>(1.0, 1.0, 1.0, 1.0);
let v_f32_zero = vec4<f32>(0.0, 0.0, 0.0, 0.0);
let v_f32_half = vec4<f32>(0.5, 0.5, 0.5, 0.5);
let v_i32_one = vec4<i32>(1, 1, 1, 1);

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

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

struct Foo {
    a: vec4<f32>,
    b: i32,
}

fn constructors() -> f32 {
    var foo: Foo;
    foo = Foo(vec4<f32>(1.0), 1);

    let mat2comp = mat2x2<f32>(
        1.0, 0.0,
        0.0, 1.0,
    );
    let mat4comp = mat4x4<f32>(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    );

    // zero value constructors
    var _ = bool();
    var _ = i32();
    var _ = u32();
    var _ = f32();
    var _ = vec2<u32>();
    var _ = mat2x2<f32>();
    var _ = array<Foo, 3>();
    var _ = Foo();

    // constructors that infer their type from their parameters
    var _ = vec2(0u);
    var _ = mat2x2(vec2(0.), vec2(0.));
    var _ = array(0, 1, 2, 3);

    return foo.a.x;
}

fn logical() {
    // unary
    let _ = !true;
    let _ = !vec2(true);

    // binary
    let _ = true || false;
    let _ = true && false;
    let _ = true | false;
    let _ = vec3(true) | vec3(false);
    let _ = true & false;
    let _ = vec4(true) & vec4(false);
}

fn arithmetic() {
    // unary
    let _ = -1;
    let _ = -1.0;
    let _ = -vec2(1);
    let _ = -vec2(1.0);

    // binary
    // Addition
    let _ = 2 + 1;
    let _ = 2u + 1u;
    let _ = 2.0 + 1.0;
    let _ = vec2(2) + vec2(1);
    let _ = vec3(2u) + vec3(1u);
    let _ = vec4(2.0) + vec4(1.0);

    // Subtraction
    let _ = 2 - 1;
    let _ = 2u - 1u;
    let _ = 2.0 - 1.0;
    let _ = vec2(2) - vec2(1);
    let _ = vec3(2u) - vec3(1u);
    let _ = vec4(2.0) - vec4(1.0);

    // Multiplication
    let _ = 2 * 1;
    let _ = 2u * 1u;
    let _ = 2.0 * 1.0;
    let _ = vec2(2) * vec2(1);
    let _ = vec3(2u) * vec3(1u);
    let _ = vec4(2.0) * vec4(1.0);

    // Division
    let _ = 2 / 1;
    let _ = 2u / 1u;
    let _ = 2.0 / 1.0;
    let _ = vec2(2) / vec2(1);
    let _ = vec3(2u) / vec3(1u);
    let _ = vec4(2.0) / vec4(1.0);

    // Remainder
    let _ = 2 % 1;
    let _ = 2u % 1u;
    let _ = 2.0 % 1.0;
    let _ = vec2(2) % vec2(1);
    let _ = vec3(2u) % vec3(1u);
    let _ = vec4(2.0) % vec4(1.0);

    // Binary arithmetic expressions with mixed scalar and vector operands
    let _ = vec2(2) + 1;
    let _ = 2 + vec2(1);

    let _ = vec2(2) - 1;
    let _ = 2 - vec2(1);

    let _ = vec2(2) * 1;
    let _ = 2 * vec2(1);

    let _ = vec2(2) / 1;
    let _ = 2 / vec2(1);

    let _ = vec2(2) % 1;
    let _ = 2 % vec2(1);

    // Matrix arithmetic
    // let _ = mat3x3<f32>() + mat3x3<f32>();
    // let _ = mat3x3<f32>() - mat3x3<f32>();

    let _ = mat3x3<f32>() * 1.0;
    let _ = 2.0 * mat3x3<f32>();

    let _ = mat4x3<f32>() * vec4(1.0);
    let _ = vec3(2.0) * mat4x3<f32>();

    let _ = mat4x3<f32>() * mat3x4<f32>();
}

fn bit() {
    // unary
    let _ = ~1;
    let _ = ~1u;
    let _ = ~vec2(1);
    let _ = ~vec3(1u);

    // binary
    let _ = 2 | 1;
    let _ = 2u | 1u;
    let _ = vec2(2) | vec2(1);
    let _ = vec3(2u) | vec3(1u);

    let _ = 2 & 1;
    let _ = 2u & 1u;
    let _ = vec2(2) & vec2(1);
    let _ = vec3(2u) & vec3(1u);

    let _ = 2 ^ 1;
    let _ = 2u ^ 1u;
    let _ = vec2(2) ^ vec2(1);
    let _ = vec3(2u) ^ vec3(1u);

    let _ = 2 << 1u;
    let _ = 2u << 1u;
    let _ = vec2(2) << vec2(1u);
    let _ = vec3(2u) << vec3(1u);

    let _ = 2 >> 1u;
    let _ = 2u >> 1u;
    let _ = vec2(2) >> vec2(1u);
    let _ = vec3(2u) >> vec3(1u);
}

fn comparison() {
    let _ = 2 == 1;
    let _ = 2u == 1u;
    let _ = 2.0 == 1.0;
    let _ = vec2(2) == vec2(1);
    let _ = vec3(2u) == vec3(1u);
    let _ = vec4(2.0) == vec4(1.0);

    let _ = 2 != 1;
    let _ = 2u != 1u;
    let _ = 2.0 != 1.0;
    let _ = vec2(2) != vec2(1);
    let _ = vec3(2u) != vec3(1u);
    let _ = vec4(2.0) != vec4(1.0);

    let _ = 2 < 1;
    let _ = 2u < 1u;
    let _ = 2.0 < 1.0;
    let _ = vec2(2) < vec2(1);
    let _ = vec3(2u) < vec3(1u);
    let _ = vec4(2.0) < vec4(1.0);

    let _ = 2 <= 1;
    let _ = 2u <= 1u;
    let _ = 2.0 <= 1.0;
    let _ = vec2(2) <= vec2(1);
    let _ = vec3(2u) <= vec3(1u);
    let _ = vec4(2.0) <= vec4(1.0);

    let _ = 2 > 1;
    let _ = 2u > 1u;
    let _ = 2.0 > 1.0;
    let _ = vec2(2) > vec2(1);
    let _ = vec3(2u) > vec3(1u);
    let _ = vec4(2.0) > vec4(1.0);

    let _ = 2 >= 1;
    let _ = 2u >= 1u;
    let _ = 2.0 >= 1.0;
    let _ = vec2(2) >= vec2(1);
    let _ = vec3(2u) >= vec3(1u);
    let _ = vec4(2.0) >= vec4(1.0);
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
}

@stage(compute) @workgroup_size(1)
fn main() {
    let _ = builtins();
    let _ = splat();
    let _ = bool_cast(v_f32_one.xyz);
    let _ = constructors();

    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
}
