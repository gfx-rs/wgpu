// An empty WGSL shader to check that task payload/mesh output
// are still properly written without being used in the shader

enable wgpu_mesh_shader;

struct TaskPayload {
    dummy: u32,
}
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}
struct PrimitiveOutput {
    @builtin(point_index) indices: u32,
}

var<task_payload> taskPayload: TaskPayload;

@task
@payload(taskPayload)
@workgroup_size(64)
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    return vec3(1, 1, 1);
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 1>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@payload(taskPayload)
@workgroup_size(64)
fn ms_main() {}
