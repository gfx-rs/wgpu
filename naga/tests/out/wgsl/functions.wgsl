fn test_fma() -> vec2<f32> {
    const a = vec2<f32>(2f, 2f);
    const b = vec2<f32>(0.5f, 0.5f);
    const c = vec2<f32>(0.5f, 0.5f);
    return fma(a, b, c);
}

fn test_integer_dot_product() -> i32 {
    const a_2_ = vec2(1i);
    const b_2_ = vec2(1i);
    let c_2_ = dot(a_2_, b_2_);
    const a_3_ = vec3(1u);
    const b_3_ = vec3(1u);
    let c_3_ = dot(a_3_, b_3_);
    let c_4_ = dot(vec4(4i), vec4(2i));
    return c_4_;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = test_fma();
    let _e1 = test_integer_dot_product();
    return;
}
