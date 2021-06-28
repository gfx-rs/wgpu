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

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    let _e0: vec4<f32> = splat();
    let _e1: i32 = unary();
    return;
}
