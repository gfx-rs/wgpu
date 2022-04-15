fn test_fma() -> vec2<f32> {
    let a = vec2<f32>(2.0, 2.0);
    let b = vec2<f32>(0.5, 0.5);
    let c = vec2<f32>(0.5, 0.5);
    return fma(a, b, c);
}

fn test_integer_dot_product() -> i32 {
    let a_2_ = vec2<i32>(1);
    let b_2_ = vec2<i32>(1);
    let c_2_ = dot(a_2_, b_2_);
    let a_3_ = vec3<u32>(1u);
    let b_3_ = vec3<u32>(1u);
    let c_3_ = dot(a_3_, b_3_);
    let c_4_ = dot(vec4<i32>(4), vec4<i32>(2));
    return c_4_;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e0 = test_fma();
    let _e1 = test_integer_dot_product();
    return;
}
