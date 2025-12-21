enable wgpu_mesh_shader;

struct TaskPayload {
    colorMask: vec4<f32>,
    visible: bool,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
    @builtin(cull_primitive) cull: bool,
    @location(1) @per_primitive colorMask: vec4<f32>,
}

struct PrimitiveInput {
    @location(1) @per_primitive colorMask: vec4<f32>,
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<task_payload> taskPayload: TaskPayload;
var<workgroup> workgroupData: f32;
var<workgroup> mesh_output: MeshOutput;

@task @payload(taskPayload) @workgroup_size(1, 1, 1) 
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    workgroupData = 1f;
    taskPayload.colorMask = vec4<f32>(1f, 1f, 0f, 1f);
    taskPayload.visible = true;
    return vec3<u32>(1u, 1u, 1u);
}

@task @payload(taskPayload) @workgroup_size(2, 1, 1) 
fn ts_divergent(@builtin(local_invocation_index) thread_id: u32) -> @builtin(mesh_task_size) vec3<u32> {
    if (thread_id == 0u) {
        taskPayload.colorMask = vec4<f32>(1f, 1f, 0f, 1f);
        taskPayload.visible = true;
        return vec3<u32>(1u, 1u, 1u);
    }
    return vec3<u32>(2u, 2u, 2u);
}

@mesh(mesh_output) @workgroup_size(1, 1, 1) @payload(taskPayload) 
fn ms_main() {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2f;
    mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
    let _e23 = taskPayload.colorMask;
    mesh_output.vertices[0].color = (vec4<f32>(0f, 1f, 0f, 1f) * _e23);
    mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
    let _e45 = taskPayload.colorMask;
    mesh_output.vertices[1].color = (vec4<f32>(0f, 0f, 1f, 1f) * _e45);
    mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
    let _e67 = taskPayload.colorMask;
    mesh_output.vertices[2].color = (vec4<f32>(1f, 0f, 0f, 1f) * _e67);
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    let _e88 = taskPayload.visible;
    mesh_output.primitives[0].cull = !(_e88);
    mesh_output.primitives[0].colorMask = vec4<f32>(1f, 0f, 1f, 1f);
    return;
}

@mesh(mesh_output) @workgroup_size(1, 1, 1) 
fn ms_no_ts() {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2f;
    mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
    mesh_output.vertices[0].color = vec4<f32>(0f, 1f, 0f, 1f);
    mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
    mesh_output.vertices[1].color = vec4<f32>(0f, 0f, 1f, 1f);
    mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
    mesh_output.vertices[2].color = vec4<f32>(1f, 0f, 0f, 1f);
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    mesh_output.primitives[0].cull = false;
    mesh_output.primitives[0].colorMask = vec4<f32>(1f, 0f, 1f, 1f);
    return;
}

@mesh(mesh_output) @workgroup_size(1, 1, 1) 
fn ms_divergent(@builtin(local_invocation_index) thread_id_1: u32) {
    if (thread_id_1 == 0u) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        workgroupData = 2f;
        mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
        mesh_output.vertices[0].color = vec4<f32>(0f, 1f, 0f, 1f);
        mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
        mesh_output.vertices[1].color = vec4<f32>(0f, 0f, 1f, 1f);
        mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
        mesh_output.vertices[2].color = vec4<f32>(1f, 0f, 0f, 1f);
        mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
        mesh_output.primitives[0].cull = false;
        mesh_output.primitives[0].colorMask = vec4<f32>(1f, 0f, 1f, 1f);
        return;
    } else {
        return;
    }
}

@fragment 
fn fs_main(vertex: VertexOutput, primitive: PrimitiveInput) -> @location(0) vec4<f32> {
    return (vertex.color * primitive.colorMask);
}
