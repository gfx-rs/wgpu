//TODO: support splatting constructors for globals?
let v_f32_one = vec4<f32>(1.0, 1.0, 1.0, 1.0);
let v_f32_zero = vec4<f32>(0.0, 0.0, 0.0, 0.0);
let v_f32_half = vec4<f32>(0.5, 0.5, 0.5, 0.5);

fn builtins() -> vec4<f32> {
    // select()
    let condition = true;
    let s1 = select(0, 1, condition);
    let s2 = select(v_f32_zero, v_f32_one, condition);
    let s3 = select(v_f32_one, v_f32_zero, vec4<bool>(false, false, false, false));
    // mix()
    let m1 = mix(v_f32_zero, v_f32_one, v_f32_half);
    let m2 = mix(v_f32_zero, v_f32_one, 0.1);
    // done
    return vec4<f32>(vec4<i32>(s1)) + s2 + m1 + m2;
}

fn splat() -> vec4<f32> {
    let a = (1.0 + vec2<f32>(2.0) - 3.0) / 4.0;
    let b = vec4<i32>(5) % 2;
    return a.xyxy + vec4<f32>(b);
}

fn unary() -> i32 {
    let a = 1;
    if (!true) { return a; } else { return ~a; };
}

struct Foo {
    a: vec4<f32>;
    b: i32;
};

fn constructors() -> f32 {
    var foo: Foo;
    foo = Foo(vec4<f32>(1.0), 1);
    return foo.a.x;
}

[[stage(compute), workgroup_size(1)]]
fn main() {
    let a = builtins();
    let b = splat();
    let c = unary();
    let d = constructors();
}
