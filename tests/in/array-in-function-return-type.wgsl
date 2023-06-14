fn ret_array() -> array<f32, 2> {
    return array<f32, 2>(1.0, 2.0);
}

@fragment
fn main() -> @location(0) vec4<f32> {
    let a = ret_array();
    return vec4<f32>(a[0], a[1], 0.0, 1.0);
}
