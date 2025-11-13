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

fn test_packed_integer_dot_product() -> u32 {
    let a_5 = 1u;
    let b_5 = 2u;
    let c_5: i32 = dot4I8Packed(a_5, b_5);

    let a_6 = 3u;
    let b_6 = 4u;
    let c_6: u32 = dot4U8Packed(a_6, b_6);

    // test baking of arguments
    let c_7: i32 = dot4I8Packed(5u + c_6, 6u + c_6);
    let c_8: u32 = dot4U8Packed(7u + c_6, 8u + c_6);
    return c_8;
}

@compute @workgroup_size(1)
fn main() {
    let a = test_fma();
    let b = test_integer_dot_product();
    let c = test_packed_integer_dot_product();
}
