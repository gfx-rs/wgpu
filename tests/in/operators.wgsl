fn splat() -> vec4<f32> {
    let a = (1.0 + vec2<f32>(2.0) - 3.0) / 4.0;
    let b = vec4<i32>(5) % 2;
    return a.xyxy + vec4<f32>(b);
}

fn unary() -> i32 {
    let a = 1;
    if (!true) { return a; } else { return ~a; };
}

[[stage(compute), workgroup_size(1)]]
fn main() {
    let a = splat();
    let b = unary();
}
