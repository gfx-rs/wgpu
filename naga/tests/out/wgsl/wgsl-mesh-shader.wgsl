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

fn helper_reader() -> bool {
    let _e2 = taskPayload.visible;
    return _e2;
}

fn helper_writer(value: bool) {
    taskPayload.visible = value;
    return;
}

@task @payload(taskPayload) @workgroup_size(64, 1, 1) 
fn ts_main(@builtin(local_invocation_id) thread_id: vec3<u32>) -> @builtin(mesh_task_size) vec3<u32> {
    if (thread_id.x == 0u) {
        taskPayload.colorMask = vec4<f32>(1f, 1f, 0f, 1f);
        helper_writer(true);
        let _e14 = helper_reader();
        taskPayload.visible = _e14;
        return vec3<u32>(1u, 1u, 1u);
    }
    return vec3<u32>(0u, 0u, 0u);
}

@mesh(mesh_output) @workgroup_size(64, 1, 1) @payload(taskPayload) 
fn ms_main(@builtin(local_invocation_id) thread_id_1: vec3<u32>) {
    if (thread_id_1.x == 0u) {
        mesh_output.vertex_count = 3u;
        mesh_output.primitive_count = 1u;
        workgroupData = 2f;
        mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
        let _e27 = taskPayload.colorMask;
        mesh_output.vertices[0].color = (vec4<f32>(0f, 1f, 0f, 1f) * _e27);
        mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
        let _e49 = taskPayload.colorMask;
        mesh_output.vertices[1].color = (vec4<f32>(0f, 0f, 1f, 1f) * _e49);
        mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
        let _e71 = taskPayload.colorMask;
        mesh_output.vertices[2].color = (vec4<f32>(1f, 0f, 0f, 1f) * _e71);
        mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
        let _e90 = helper_reader();
        mesh_output.primitives[0].cull = !(_e90);
        mesh_output.primitives[0].colorMask = vec4<f32>(1f, 0f, 1f, 1f);
        return;
    } else {
        return;
    }
}

@mesh(mesh_output) @workgroup_size(64, 1, 1) 
fn ms_no_ts(@builtin(local_invocation_id) thread_id_2: vec3<u32>) {
    if (thread_id_2.x == 0u) {
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
