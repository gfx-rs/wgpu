struct Foo {
    a: vec4<f32>;
    b: i32;
};

let v_f32_one: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);
let v_f32_zero: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
let v_f32_half: vec4<f32> = vec4<f32>(0.5, 0.5, 0.5, 0.5);
fn builtins() -> vec4<f32> {
    let s1_: i32 = select(0, 1, true);
    let s2_: vec4<f32> = select(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), true);
    let s3_: vec4<f32> = select(vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<bool>(false, false, false, false));
    let m1_: vec4<f32> = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(0.5, 0.5, 0.5, 0.5));
    let m2_: vec4<f32> = mix(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), 0.1);
    return (((vec4<f32>(vec4<i32>(s1_)) + s2_) + m1_) + m2_);
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

fn constructors() -> f32 {
    var foo: Foo;

    foo = Foo(vec4<f32>(1.0), 1);
    let _e9: vec4<f32> = foo.a;
    return _e9.x;
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    let _e3: vec4<f32> = builtins();
    let _e4: vec4<f32> = splat();
    let _e5: i32 = unary();
    let _e6: f32 = constructors();
    return;
}
