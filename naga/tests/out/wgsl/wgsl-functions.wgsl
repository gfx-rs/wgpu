fn test_fma() -> vec2<f32> {
    let a = vec2<f32>(2f, 2f);
    let b = vec2<f32>(0.5f, 0.5f);
    let c = vec2<f32>(0.5f, 0.5f);
    return fma(a, b, c);
}

fn test_integer_dot_product() -> i32 {
    let a_2_ = vec2(1i);
    let b_2_ = vec2(1i);
    let c_2_ = dot(a_2_, b_2_);
    let a_3_ = vec3(1u);
    let b_3_ = vec3(1u);
    let c_3_ = dot(a_3_, b_3_);
    let c_4_ = dot(vec4(4i), vec4(2i));
    return c_4_;
}

fn test_packed_integer_dot_product() -> u32 {
    let c_5_ = dot4I8Packed(1u, 2u);
    let c_6_ = dot4U8Packed(3u, 4u);
    let c_7_ = dot4I8Packed((5u + c_6_), (6u + c_6_));
    let c_8_ = dot4U8Packed((7u + c_6_), (8u + c_6_));
    return c_8_;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = test_fma();
    let _e1 = test_integer_dot_product();
    let _e2 = test_packed_integer_dot_product();
    return;
}
