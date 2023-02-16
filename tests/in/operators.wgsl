//TODO: support splatting constructors for globals?
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
    _ = bool();
    _ = i32();
    _ = u32();
    _ = f32();
    _ = vec2<u32>();
    _ = mat2x2<f32>();
    _ = array<Foo, 3>();
    _ = Foo();

    // constructors that infer their type from their parameters
    _ = vec2(0u);
    _ = mat2x2(vec2(0.), vec2(0.));
    _ = array(0, 1, 2, 3);

    // identity constructors
    _ = bool(bool());
    _ = i32(i32());
    _ = u32(u32());
    _ = f32(f32());
    _ = vec2<u32>(vec2<u32>());
    _ = mat2x3<f32>(mat2x3<f32>());
    _ = vec2(vec2<u32>());
    _ = mat2x3(mat2x3<f32>());

    return foo.a.x;
}

fn logical() {
    // unary
    _ = !true;
    _ = !vec2(true);

    // binary
    _ = true || false;
    _ = true && false;
    _ = true | false;
    _ = vec3(true) | vec3(false);
    _ = true & false;
    _ = vec4(true) & vec4(false);
}

fn arithmetic() {
    // unary
    _ = -1.0;
    _ = -vec2(1);
    _ = -vec2(1.0);

    // binary
    // Addition
    _ = 2 + 1;
    _ = 2u + 1u;
    _ = 2.0 + 1.0;
    _ = vec2(2) + vec2(1);
    _ = vec3(2u) + vec3(1u);
    _ = vec4(2.0) + vec4(1.0);

    // Subtraction
    _ = 2 - 1;
    _ = 2u - 1u;
    _ = 2.0 - 1.0;
    _ = vec2(2) - vec2(1);
    _ = vec3(2u) - vec3(1u);
    _ = vec4(2.0) - vec4(1.0);

    // Multiplication
    _ = 2 * 1;
    _ = 2u * 1u;
    _ = 2.0 * 1.0;
    _ = vec2(2) * vec2(1);
    _ = vec3(2u) * vec3(1u);
    _ = vec4(2.0) * vec4(1.0);

    // Division
    _ = 2 / 1;
    _ = 2u / 1u;
    _ = 2.0 / 1.0;
    _ = vec2(2) / vec2(1);
    _ = vec3(2u) / vec3(1u);
    _ = vec4(2.0) / vec4(1.0);

    // Remainder
    _ = 2 % 1;
    _ = 2u % 1u;
    _ = 2.0 % 1.0;
    _ = vec2(2) % vec2(1);
    _ = vec3(2u) % vec3(1u);
    _ = vec4(2.0) % vec4(1.0);

    // Binary arithmetic expressions with mixed scalar and vector operands
    _ = vec2(2) + 1;
    _ = 2 + vec2(1);
    _ = vec2(2u) + 1u;
    _ = 2u + vec2(1u);
    _ = vec2(2.0) + 1.0;
    _ = 2.0 + vec2(1.0);

    _ = vec2(2) - 1;
    _ = 2 - vec2(1);
    _ = vec2(2u) - 1u;
    _ = 2u - vec2(1u);
    _ = vec2(2.0) - 1.0;
    _ = 2.0 - vec2(1.0);

    _ = vec2(2) * 1;
    _ = 2 * vec2(1);
    _ = vec2(2u) * 1u;
    _ = 2u * vec2(1u);
    _ = vec2(2.0) * 1.0;
    _ = 2.0 * vec2(1.0);

    _ = vec2(2) / 1;
    _ = 2 / vec2(1);
    _ = vec2(2u) / 1u;
    _ = 2u / vec2(1u);
    _ = vec2(2.0) / 1.0;
    _ = 2.0 / vec2(1.0);

    _ = vec2(2) % 1;
    _ = 2 % vec2(1);
    _ = vec2(2u) % 1u;
    _ = 2u % vec2(1u);
    _ = vec2(2.0) % 1.0;
    _ = 2.0 % vec2(1.0);

    // Matrix arithmetic
    _ = mat3x3<f32>() + mat3x3<f32>();
    _ = mat3x3<f32>() - mat3x3<f32>();

    _ = mat3x3<f32>() * 1.0;
    _ = 2.0 * mat3x3<f32>();

    _ = mat4x3<f32>() * vec4(1.0);
    _ = vec3f(2.0) * mat4x3f();

    _ = mat4x3<f32>() * mat3x4<f32>();
}

fn bit() {
    // unary
    _ = ~1;
    _ = ~1u;
    _ = ~vec2(1);
    _ = ~vec3(1u);

    // binary
    _ = 2 | 1;
    _ = 2u | 1u;
    _ = vec2(2) | vec2(1);
    _ = vec3(2u) | vec3(1u);

    _ = 2 & 1;
    _ = 2u & 1u;
    _ = vec2(2) & vec2(1);
    _ = vec3(2u) & vec3(1u);

    _ = 2 ^ 1;
    _ = 2u ^ 1u;
    _ = vec2(2) ^ vec2(1);
    _ = vec3(2u) ^ vec3(1u);

    _ = 2 << 1u;
    _ = 2u << 1u;
    _ = vec2(2) << vec2(1u);
    _ = vec3(2u) << vec3(1u);

    _ = 2 >> 1u;
    _ = 2u >> 1u;
    _ = vec2(2) >> vec2(1u);
    _ = vec3(2u) >> vec3(1u);
}

fn comparison() {
    _ = 2 == 1;
    _ = 2u == 1u;
    _ = 2.0 == 1.0;
    _ = vec2(2) == vec2(1);
    _ = vec3(2u) == vec3(1u);
    _ = vec4(2.0) == vec4(1.0);

    _ = 2 != 1;
    _ = 2u != 1u;
    _ = 2.0 != 1.0;
    _ = vec2(2) != vec2(1);
    _ = vec3(2u) != vec3(1u);
    _ = vec4(2.0) != vec4(1.0);

    _ = 2 < 1;
    _ = 2u < 1u;
    _ = 2.0 < 1.0;
    _ = vec2(2) < vec2(1);
    _ = vec3(2u) < vec3(1u);
    _ = vec4(2.0) < vec4(1.0);

    _ = 2 <= 1;
    _ = 2u <= 1u;
    _ = 2.0 <= 1.0;
    _ = vec2(2) <= vec2(1);
    _ = vec3(2u) <= vec3(1u);
    _ = vec4(2.0) <= vec4(1.0);

    _ = 2 > 1;
    _ = 2u > 1u;
    _ = 2.0 > 1.0;
    _ = vec2(2) > vec2(1);
    _ = vec3(2u) > vec3(1u);
    _ = vec4(2.0) > vec4(1.0);

    _ = 2 >= 1;
    _ = 2u >= 1u;
    _ = 2.0 >= 1.0;
    _ = vec2(2) >= vec2(1);
    _ = vec3(2u) >= vec3(1u);
    _ = vec4(2.0) >= vec4(1.0);
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
    _ = builtins();
    _ = splat();
    _ = bool_cast(v_f32_one.xyz);
    _ = constructors();

    logical();
    arithmetic();
    bit();
    comparison();
    assignment();
}

fn negation_avoids_prefix_decrement() {
    _ = -1;
    _ = - -2;
    _ = -(-3);
    _ = -(- 4);
    _ = - - -5;
    _ = - - - - 6;
    _ = - - -(- -7);
    _ = (- - - - -8);
}
