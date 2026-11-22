enable wgpu_mesh_shader;
requires unrestricted_pointer_parameters;

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

var<task_payload> taskPayload: TaskPayload;

fn fill(p: ptr<task_payload, TaskPayload>) {
    p.colors[0] = vec4(1.0, 0.0, 0.0, 1.0);
    p.colors[1] = vec4(0.0, 1.0, 0.0, 1.0);
    p.visible = true;
}

fn color(p: ptr<task_payload, vec4<f32>>) -> vec4<f32> {
    return *p;
}

@task
@payload(taskPayload)
@workgroup_size(1)
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    fill(&taskPayload);
    return vec3(1, 1, 1);
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@payload(taskPayload)
@workgroup_size(1)
fn ms_main() {
    mesh_output.vertex_count = 3;
    mesh_output.primitive_count = 1;
    mesh_output.vertices[0].position = vec4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices[0].color = color(&taskPayload.colors[0]);
    mesh_output.vertices[1].position = vec4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices[1].color = color(&taskPayload.colors[1]);
    mesh_output.vertices[2].position = vec4(1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices[2].color = color(&taskPayload.colors[0]);
    mesh_output.primitives[0].indices = vec3(0u, 1u, 2u);
}
