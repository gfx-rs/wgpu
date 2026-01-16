enable primitive_index;

@fragment 
fn func(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(index), 1f, 1f, 1f);
}
