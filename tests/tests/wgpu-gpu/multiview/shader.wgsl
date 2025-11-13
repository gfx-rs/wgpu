const triangles = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    return vec4f(triangles[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(view_index) view_index: u32) -> @location(0) vec4f {
    return vec4f(f32(view_index) * 0.25 + 0.125);
}