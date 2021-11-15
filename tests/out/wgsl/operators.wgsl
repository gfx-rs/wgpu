struct Foo {
    a: vec4<f32>;
    b: i32;
};

let v_f32_one: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
let v_f32_zero: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
let v_f32_half: vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.5);
let v_i32_one: vec4<i32> = vec4<i32>(1, 1, 1, 1);
fn builtins() -> vec4<f32> {
    let s1_: i32 = select(0, 1, true);
    let s2_: vec4<f32> = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), true);
    let s3_: vec4<f32> = select(vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<bool>(false, false, false, false));
    let m1_: vec4<f32> = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.5, 0.5, 0.5, 0.5));
    let m2_: vec4<f32> = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), 0.10000000149011612);
    let b1_: f32 = bitcast<f32>(vec4<i32>(1, 1, 1, 1).x);
    let b2_: vec4<f32> = bitcast<vec4<f32>>(vec4<i32>(1, 1, 1, 1));
    let v_i32_zero: vec4<i32> = vec4<i32>(vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return (((((vec4<f32>((vec4<i32>(s1_) + v_i32_zero)) + s2_) + m1_) + m2_) + vec4<f32>(b1_)) + b2_);
}

fn splat() -> vec4<f32> {
    let a: vec2<f32> = (((vec2<f32>(1.0) + vec2<f32>(2.0)) - vec2<f32>(3.0)) / vec2<f32>(4.0));
    let b: vec4<i32> = (vec4<i32>(5) % vec4<i32>(2));
    return (a.xyxy + vec4<f32>(b));
}

fn unary() -> i32 {
    if (!(true)) {
        return 1;
    } else {
        return ~(1);
    }
}

fn bool_cast(x: vec3<f32>) -> vec3<f32> {
    let y: vec3<bool> = vec3<bool>(x);
    return vec3<f32>(y);
}

fn constructors() -> f32 {
    var foo: Foo;

    foo = Foo(vec4<f32>(1.0), 1);
    let e11: f32 = foo.a.x;
    return e11;
}

fn modulo() {
    let a_1: i32 = (1 % 1);
    let b_1: f32 = (1.0 % 1.0);
    let c: vec3<i32> = (vec3<i32>(1) % vec3<i32>(1));
    let d: vec3<f32> = (vec3<f32>(1.0) % vec3<f32>(1.0));
}

fn scalar_times_matrix() {
    let model: mat4x4<f32> = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let assertion: mat4x4<f32> = (2.0 * model);
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    let e4: vec4<f32> = builtins();
    let e5: vec4<f32> = splat();
    let e6: i32 = unary();
    let e8: vec3<f32> = bool_cast(vec4<f32>(1.0, 1.0, 1.0, 1.0).xyz);
    let e9: f32 = constructors();
    modulo();
    scalar_times_matrix();
    return;
}
