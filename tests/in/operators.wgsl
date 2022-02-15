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

fn unary() -> i32 {
    let a = 1;
    if !true { return a; } else { return ~a; };
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y = vec3<bool>(x);
    return vec3<f32>(y);
}

struct Foo {
    a: vec4<f32>;
    b: i32;
};

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

    return foo.a.x;
}

fn modulo() {
    // Modulo operator on float scalar or vector must be converted to mod function for GLSL
    let a = 1 % 1;
    let b = 1.0 % 1.0;
    let c = vec3<i32>(1) % vec3<i32>(1);
    let d = vec3<f32>(1.0) % vec3<f32>(1.0);
}

fn scalar_times_matrix() {
    let model = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );

    let assertion: mat4x4<f32> = 2.0 * model;
}

fn logical() {
    let a = true | false;
    let b = true & false;
}

fn binary_assignment() {
    var a = 1;
    a += 1;
    a -= 1;
    a *= a;
    a /= a;
    a %= 1;
    a ^= 0;
    a &= 0;
}

@stage(compute) @workgroup_size(1)
fn main() {
    let a = builtins();
    let b = splat();
    let c = unary();
    let d = bool_cast(v_f32_one.xyz);
    let e = constructors();
    modulo();
    scalar_times_matrix();
    logical();
    binary_assignment();
}
