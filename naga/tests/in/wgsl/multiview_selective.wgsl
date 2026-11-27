struct FragmentIn {
    @builtin(position) position: vec4f,
    @builtin(view_index) view_index: u32,
}

@vertex
fn vs_main(@builtin(view_index) view_index: u32) -> @builtin(position) vec4f {
    return vec4f(f32(view_index));
}

@fragment
fn fs_main(in: FragmentIn) -> @location(0) vec4f {
    return vec4f(f32(in.view_index));
}
