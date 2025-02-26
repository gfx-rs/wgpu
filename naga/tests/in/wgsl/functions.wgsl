fn test_fma() -> vec2<f32> {
    let a = vec2<f32>(2.0, 2.0);
    let b = vec2<f32>(0.5, 0.5);
    let c = vec2<f32>(0.5, 0.5);

    // Hazard: HLSL needs a different intrinsic function for f32 and f64
    // See: https://github.com/gfx-rs/naga/issues/1579
    return fma(a, b, c);
}

fn test_integer_dot_product() -> i32 {
    let a_2 = vec2<i32>(1);
    let b_2 = vec2<i32>(1);
    let c_2: i32 = dot(a_2, b_2);

    let a_3 = vec3<u32>(1u);
    let b_3 = vec3<u32>(1u);
    let c_3: u32 = dot(a_3, b_3);

    // test baking of arguments
    let c_4: i32 = dot(vec4<i32>(4), vec4<i32>(2));
    return c_4;
}

@compute @workgroup_size(1)
fn main() {
    let a = test_fma();
    let b = test_integer_dot_product();
}
