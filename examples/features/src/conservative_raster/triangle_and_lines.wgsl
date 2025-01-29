@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let i: i32 = i32(vertex_index % 3u);
    let x: f32 = f32(i - 1) * 0.75;
    let y: f32 = f32((i & 1) * 2 - 1) * 0.75 + x * 0.2 + 0.1;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main_red() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main_blue() -> @location(0) vec4<f32> {
    return vec4<f32>(0.13, 0.31, 0.85, 1.0); // cornflower blue in linear space
}

@fragment
fn fs_main_white() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
