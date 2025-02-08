fn ret_array() -> array<f32, 2> {
    return array<f32, 2>(1f, 2f);
}

fn ret_array_array() -> array<array<f32, 2>, 3> {
    let _e0 = ret_array();
    let _e1 = ret_array();
    let _e2 = ret_array();
    return array<array<f32, 2>, 3>(_e0, _e1, _e2);
}

@fragment 
fn main() -> @location(0) vec4<f32> {
    let _e0 = ret_array_array();
    return vec4<f32>(_e0[0][0], _e0[0][1], 0f, 1f);
}
