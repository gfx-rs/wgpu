enable wgpu_mesh_shader;

struct TaskPayload {
    visible: array<u32, 64>,
    count: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 64>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 126>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<task_payload> payload: TaskPayload;
var<workgroup> scratch: array<vec4<f32>, 256>;
var<workgroup> mesh_output: MeshOutput;

@task
@payload(payload)
@workgroup_size(32)
fn ts_main(@builtin(local_invocation_index) index: u32) -> @builtin(mesh_task_size) vec3<u32> {
    scratch[index] = vec4(f32(index));
    payload.visible[index] = u32(scratch[index].x);
    payload.count = 32u;
    return vec3(1, 1, 1);
}

@mesh(mesh_output)
@payload(payload)
@workgroup_size(32)
fn ms_main(@builtin(local_invocation_id) id: vec3<u32>) {
    mesh_output.vertex_count = 3;
    mesh_output.primitive_count = 1;
    mesh_output.vertices[id.x].position = vec4(f32(payload.visible[id.x]));
    mesh_output.primitives[0].indices = vec3<u32>(0, 1, 2);
}
