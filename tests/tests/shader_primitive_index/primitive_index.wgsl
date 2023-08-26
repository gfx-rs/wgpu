@vertex
fn vs_main(@location(0) xy: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(xy, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> {
    if ((index % 2u) == 0u) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    } else {
        return vec4<f32>(0.0, 0.0, 1.0, 1.0);
    }
}
