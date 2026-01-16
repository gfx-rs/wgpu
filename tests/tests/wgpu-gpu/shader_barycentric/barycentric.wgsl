@vertex
fn vs_main(@location(0) xy: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(xy, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(barycentric) bary: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(bary * 1.1 - 0.05, 1.0);
}

@fragment
fn fs_main_no_perspective(@builtin(barycentric_no_perspective) bary: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(bary * 1.1 - 0.05, 1.0);
}
