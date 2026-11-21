// Task and mesh shaders: task payload size, mesh output limits, and per-primitive outputs.
enable wgpu_mesh_shader;

struct TaskPayload {
    color: vec4<f32>,
    visible: bool,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
    @builtin(cull_primitive) cull: bool,
    @per_primitive @location(1) mask: vec4<f32>,
}

struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<task_payload> payload: TaskPayload;
var<workgroup> mesh_output: MeshOutput;

@task
@payload(payload)
@workgroup_size(1)
fn ts_main() -> @builtin(mesh_task_size) vec3<u32> {
    payload.color = vec4(1.0, 1.0, 0.0, 1.0);
    payload.visible = true;
    return vec3(1, 1, 1);
}

@mesh(mesh_output)
@payload(payload)
@workgroup_size(1)
fn ms_main() {
    mesh_output.vertex_count = 3;
    mesh_output.primitive_count = 1;
    mesh_output.vertices[0].position = vec4(0.0, 1.0, 0.0, 1.0);
    mesh_output.vertices[0].color = payload.color;
    mesh_output.vertices[1].position = vec4(-1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices[1].color = payload.color;
    mesh_output.vertices[2].position = vec4(1.0, -1.0, 0.0, 1.0);
    mesh_output.vertices[2].color = payload.color;
    mesh_output.primitives[0].indices = vec3<u32>(0, 1, 2);
    mesh_output.primitives[0].cull = !payload.visible;
    mesh_output.primitives[0].mask = vec4(1.0);
}

struct FragmentInput {
    @location(0) color: vec4<f32>,
    @per_primitive @location(1) mask: vec4<f32>,
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return in.color * in.mask;
}
