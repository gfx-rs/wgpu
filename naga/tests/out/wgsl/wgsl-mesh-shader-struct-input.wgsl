enable wgpu_mesh_shader;

struct TaskPayload {
    group: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}

struct TaskInput {
    @builtin(local_invocation_index) index: u32,
    @builtin(workgroup_id) group: vec3<u32>,
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

struct MeshInput {
    @builtin(local_invocation_index) index: u32,
    @builtin(workgroup_id) group: vec3<u32>,
}

var<task_payload> taskPayload: TaskPayload;
var<workgroup> mesh_output: MeshOutput;

@task @payload(taskPayload) @workgroup_size(64, 1, 1)
fn ts_main(input: TaskInput, @builtin(subgroup_size) subgroup_size: u32) -> @builtin(mesh_task_size) vec3<u32> {
    if (input.index == 0u) {
        taskPayload.group = (input.group.x + subgroup_size);
    }
    return vec3<u32>(1u, 1u, 1u);
}

@task @payload(taskPayload) @workgroup_size(64, 1, 1)
fn ts_index(@builtin(local_invocation_index) index: u32) -> @builtin(mesh_task_size) vec3<u32> {
    if (index == 0u) {
        taskPayload.group = 1u;
    }
    return vec3<u32>(1u, 1u, 1u);
}

@mesh(mesh_output) @workgroup_size(64, 1, 1) @payload(taskPayload)
fn ms_main(input_1: MeshInput, @builtin(subgroup_invocation_id) lane: u32) {
    if (input_1.index == 0u) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    }
    if (input_1.index < 3u) {
        let _e29 = taskPayload.group;
        mesh_output.vertices[input_1.index].position = vec4<f32>(f32(((_e29 + input_1.group.x) + lane)), 0f, 0f, 1f);
        return;
    } else {
        return;
    }
}
