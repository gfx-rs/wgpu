
@vertex
fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
    return vec4f(position, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(view_index) view_index: u32) -> @location(0) vec4f {
    return vec4f(f32(view_index) * 0.25 + 0.125);
}