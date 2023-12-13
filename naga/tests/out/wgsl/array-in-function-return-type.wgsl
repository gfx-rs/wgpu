fn ret_array() -> array<f32, 2> {
    return array<f32, 2>(1f, 2f);
}

@fragment 
fn main() -> @location(0) vec4<f32> {
    let _e0 = ret_array();
    return vec4<f32>(_e0[0], _e0[1], 0f, 1f);
}
