// Task and mesh entry points with subgroup built-in inputs, which the HLSL
// backend gathers into an input struct.

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

var<task_payload> taskPayload: TaskPayload;

struct TaskInput {
    @builtin(local_invocation_index) index: u32,
    @builtin(workgroup_id) group: vec3<u32>,
}

@task
@payload(taskPayload)
@workgroup_size(64)
fn ts_main(input: TaskInput, @builtin(subgroup_size) subgroup_size: u32) -> @builtin(mesh_task_size) vec3<u32> {
    if input.index == 0u {
        taskPayload.group = input.group.x + subgroup_size;
    }
    return vec3(1, 1, 1);
}

// The local invocation index is an argument of its own.
@task
@payload(taskPayload)
@workgroup_size(64)
fn ts_index(@builtin(local_invocation_index) index: u32) -> @builtin(mesh_task_size) vec3<u32> {
    if index == 0u {
        taskPayload.group = 1u;
    }
    return vec3(1, 1, 1);
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_output: MeshOutput;

struct MeshInput {
    @builtin(local_invocation_index) index: u32,
    @builtin(workgroup_id) group: vec3<u32>,
}

@mesh(mesh_output)
@payload(taskPayload)
@workgroup_size(64)
fn ms_main(input: MeshInput, @builtin(subgroup_invocation_id) lane: u32) {
    if input.index == 0u {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        mesh_output.primitives[0].indices = vec3(0u, 1u, 2u);
    }
    if input.index < 3u {
        mesh_output.vertices[input.index].position = vec4(f32(taskPayload.group + input.group.x + lane), 0.0, 0.0, 1.0);
    }
}
