enable wgpu_mesh_shader;
enable primitive_index;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
    @builtin(primitive_index) index: u32,
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output) @workgroup_size(1, 1, 1)
fn ms_main() {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
    mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
    mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    mesh_output.primitives[0].index = 7u;
    return;
}

@fragment
fn fs_main(@builtin(primitive_index) index: u32) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(index), 1f, 1f, 1f);
}
