enable wgpu_per_vertex;

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @interpolate(flat) @location(0) z: f32,
}

@vertex
fn vs_main(@location(0) xyz: vec3<f32>) -> VertexOutput {
    return VertexOutput(vec4<f32>(xyz.xy, 0.0, 1.0), xyz.z);
}

@fragment
fn fs_main(@interpolate(per_vertex) @location(0) z: array<f32, 3>) -> @location(0) vec4<f32> {
    return vec4(z[0], z[1], z[2], 1.0);
}
