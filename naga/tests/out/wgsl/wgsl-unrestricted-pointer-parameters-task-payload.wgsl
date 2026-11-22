enable wgpu_mesh_shader;

struct TaskPayload {
    colors: array<vec4<f32>, 2>,
    visible: bool,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
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

var<task_payload> taskPayload: TaskPayload;
var<workgroup> mesh_output: MeshOutput;

fn fill(p: ptr<task_payload, TaskPayload>) {
    (*p).colors[0] = vec4<f32>(1f, 0f, 0f, 1f);
    (*p).colors[1] = vec4<f32>(0f, 1f, 0f, 1f);
    (*p).visible = true;
    return;
}

fn color(p_1: ptr<task_payload, vec4<f32>>) -> vec4<f32> {
    let _e1 = (*p_1);
    return _e1;
}

@task @payload(taskPayload) @workgroup_size(1, 1, 1)
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    fill((&taskPayload));
    return vec3<u32>(1u, 1u, 1u);
}

@mesh(mesh_output) @workgroup_size(1, 1, 1) @payload(taskPayload)
fn ms_main() {
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    mesh_output.vertices[0].position = vec4<f32>(0f, 1f, 0f, 1f);
    let _e22 = color((&taskPayload.colors[0]));
    mesh_output.vertices[0].color = _e22;
    mesh_output.vertices[1].position = vec4<f32>(-1f, -1f, 0f, 1f);
    let _e39 = color((&taskPayload.colors[1]));
    mesh_output.vertices[1].color = _e39;
    mesh_output.vertices[2].position = vec4<f32>(1f, -1f, 0f, 1f);
    let _e56 = color((&taskPayload.colors[0]));
    mesh_output.vertices[2].color = _e56;
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    return;
}
