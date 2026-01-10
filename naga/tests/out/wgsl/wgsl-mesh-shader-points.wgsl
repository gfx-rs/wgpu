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

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 1>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<task_payload> taskPayload: TaskPayload;
var<workgroup> mesh_output: MeshOutput;

@task @payload(taskPayload) @workgroup_size(1, 1, 1) 
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    return vec3<u32>(1u, 1u, 1u);
}

@mesh(mesh_output) @workgroup_size(1, 1, 1) @payload(taskPayload) 
fn ms_main() {
    return;
}
