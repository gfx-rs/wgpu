fn test_fma() -> vec2<f32> {
    let a = vec2<f32>(2.0, 2.0);
    let b = vec2<f32>(0.5, 0.5);
    let c = vec2<f32>(0.5, 0.5);

    // Hazard: HLSL needs a different intrinsic function for f32 and f64
    // See: https://github.com/gfx-rs/naga/issues/1579
    return fma(a, b, c);
}


[[stage(compute), workgroup_size(1)]]
fn main() {
    let a = test_fma();
}
