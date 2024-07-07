// Out of order to test sorting.
struct FragmentIn {
    @location(1) value: f32,
    @location(3) value2: f32,
    @builtin(position) position: vec4<f32>,
    // @location(0) unused_value: f32,
    // @location(2) unused_value2: vec4<f32>,
}

@fragment
fn fs_main(v_out: FragmentIn) -> @location(0) vec4<f32> {
    return vec4<f32>(v_out.value, v_out.value, v_out.value2, v_out.value2);
}
