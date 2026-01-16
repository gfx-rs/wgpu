@fragment
fn fs_main(@location(0) @interpolate(per_vertex) v: array<f32, 3>) -> @location(0) vec4<f32> {
    return vec4(v[0], v[1], v[2], 1.0);
}
