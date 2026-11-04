enable clip_distances;
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @builtin(clip_distances) clip_distances: array<f32, 2>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
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
    mesh_output.vertices[0].clip_distances[0] = 0.5f;
    mesh_output.vertices[0].clip_distances[1] = -0.5f;
    mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
    mesh_output.vertices[1].clip_distances[0] = 0.5f;
    mesh_output.vertices[1].clip_distances[1] = -0.5f;
    mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
    mesh_output.vertices[2].clip_distances[0] = 0.5f;
    mesh_output.vertices[2].clip_distances[1] = -0.5f;
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    return;
}
