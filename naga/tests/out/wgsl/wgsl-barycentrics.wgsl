@fragment 
fn fs_main(@builtin(barycentric) bary: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(bary, 1f);
}
