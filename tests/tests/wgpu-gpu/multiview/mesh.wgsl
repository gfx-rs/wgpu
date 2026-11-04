enable wgpu_mesh_shader;

const triangles = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) value: f32,
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

@mesh(mesh_output)
@workgroup_size(1)
fn ms_main(@builtin(view_index) view_index: u32) {
    mesh_output.vertex_count = 3;
    mesh_output.primitive_count = 1;
    for (var i = 0u; i < 3u; i++) {
        mesh_output.vertices[i].position = vec4f(triangles[i], 0.0, 1.0);
        mesh_output.vertices[i].value = f32(view_index) * 0.25 + 0.125;
    }
    mesh_output.primitives[0].indices = vec3<u32>(0, 1, 2);
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4f {
    return vec4f(vertex.value);
}
