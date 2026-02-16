@fragment
fn fs_main(@builtin(barycentric) bary: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4(bary, 1.0);
}

@fragment
fn fs_main_no_perspective(@builtin(barycentric_no_perspective) bary: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4(bary, 1.0);
}
