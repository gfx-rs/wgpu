@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(in_vertex_index & 3u) - 1);
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1);

    return vec4<f32>(x, y, f32(in_vertex_index & 4u) / 8.0, 1.0);
}
