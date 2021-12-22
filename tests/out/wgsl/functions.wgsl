fn test_fma() -> vec2<f32> {
    let a = vec2<f32>(2.0, 2.0);
    let b = vec2<f32>(0.5, 0.5);
    let c = vec2<f32>(0.5, 0.5);
    return fma(a, b, c);
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    let _e0 = test_fma();
    return;
}
