enable mesh_shading;

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
    @per_primitive @location(1) colorMask: vec4<f32>,
}

struct PrimitiveInput {
    @per_primitive @location(1) colorMask: vec4<f32>,
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

@mesh(mesh_output)@payload(taskPayload) @workgroup_size(1, 1, 1) 
fn ms_main(@builtin(local_invocation_index) index: u32, @builtin(global_invocation_id) id: vec3<u32>) {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    workgroupData = 2f;
    mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
    let _e25 = taskPayload.colorMask;
    mesh_output.vertices[0].color = (vec4<f32>(0f, 1f, 0f, 1f) * _e25);
    mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
    let _e47 = taskPayload.colorMask;
    mesh_output.vertices[1].color = (vec4<f32>(0f, 0f, 1f, 1f) * _e47);
    mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
    let _e69 = taskPayload.colorMask;
    mesh_output.vertices[2].color = (vec4<f32>(1f, 0f, 0f, 1f) * _e69);
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    let _e90 = taskPayload.visible;
    mesh_output.primitives[0].cull = !(_e90);
    mesh_output.primitives[0].colorMask = vec4<f32>(1f, 0f, 1f, 1f);
    return;
}

@fragment 
fn fs_main(vertex: VertexOutput, primitive: PrimitiveInput) -> @location(0) vec4<f32> {
    return (vertex.color * primitive.colorMask);
}
