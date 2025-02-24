fn ret_array() -> array<f32, 2> {
    return array<f32, 2>(1.0, 2.0);
}

fn ret_array_array() -> array<array<f32, 2>, 3> {
    return array<array<f32, 2>, 3>(ret_array(), ret_array(), ret_array());
}

@fragment
fn main() -> @location(0) vec4<f32> {
    let a = ret_array_array();
    return vec4<f32>(a[0][0], a[0][1], 0.0, 1.0);
}
