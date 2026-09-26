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

@task @payload(payload) @workgroup_size(32, 1, 1)
fn ts_main(@builtin(local_invocation_index) index: u32) -> @builtin(mesh_task_size) vec3<u32> {
    var zero_init_index: u32;

    zero_init_index = index;
    loop {
        let _e23 = zero_init_index;
        payload.visible[_e23] = u32();
        continuing {
            let _e30 = zero_init_index;
            let _e31 = (_e30 + 32u);
            zero_init_index = _e31;
            break if (_e31 >= 64u);
        }
    }
    zero_init_index = index;
    loop {
        let _e39 = zero_init_index;
        scratch[_e39] = vec4<f32>();
        continuing {
            let _e45 = zero_init_index;
            let _e46 = (_e45 + 32u);
            zero_init_index = _e46;
            break if (_e46 >= 256u);
        }
    }
    if (index == 0u) {
        payload.count = u32();
    }
    workgroupBarrier();
    scratch[index] = vec4(f32(index));
    let _e11 = scratch[index].x;
    payload.visible[index] = u32(_e11);
    payload.count = 32u;
    return vec3<u32>(1u, 1u, 1u);
}

@mesh(mesh_output) @workgroup_size(32, 1, 1) @payload(payload)
fn ms_main(@builtin(local_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local_invocation_index: u32) {
    var zero_init_index_1: u32;

    zero_init_index_1 = local_invocation_index;
    loop {
        let _e30 = zero_init_index_1;
        mesh_output.vertices[_e30] = VertexOutput();
        continuing {
            let _e37 = zero_init_index_1;
            let _e38 = (_e37 + 32u);
            zero_init_index_1 = _e38;
            break if (_e38 >= 64u);
        }
    }
    zero_init_index_1 = local_invocation_index;
    loop {
        let _e42 = zero_init_index_1;
        mesh_output.primitives[_e42] = PrimitiveOutput();
        continuing {
            let _e49 = zero_init_index_1;
            let _e50 = (_e49 + 32u);
            zero_init_index_1 = _e50;
            break if (_e50 >= 126u);
        }
    }
    if (local_invocation_index == 0u) {
        mesh_output.vertex_count = u32();
        mesh_output.primitive_count = u32();
    }
    workgroupBarrier();
    mesh_output.vertex_count = 3u;
    mesh_output.primitive_count = 1u;
    let _e16 = payload.visible[id.x];
    mesh_output.vertices[id.x].position = vec4(f32(_e16));
    mesh_output.primitives[0].indices = vec3<u32>(0u, 1u, 2u);
    return;
}
