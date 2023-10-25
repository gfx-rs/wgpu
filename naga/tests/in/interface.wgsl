// Testing various parts of the pipeline interface: locations, built-ins, and entry points

struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(1) _varying: f32,
}

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @location(10) color: u32,
) -> VertexOutput {
    let tmp = vertex_index + instance_index + color;
    return VertexOutput(vec4<f32>(1.0), f32(tmp));
}

struct FragmentOutput {
    @builtin(frag_depth) depth: f32,
    @builtin(sample_mask) sample_mask: u32,
    @location(0) color: f32,
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) front_facing: bool,
    @builtin(sample_index) sample_index: u32,
    @builtin(sample_mask) sample_mask: u32,
) -> FragmentOutput {
    let mask = sample_mask & (1u << sample_index);
    let color = select(0.0, 1.0, front_facing);
    return FragmentOutput(in._varying, mask, color);
}

var<workgroup> output: array<u32, 1>;

@compute @workgroup_size(1)
fn compute(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>,
) {
    output[0] = global_id.x + local_id.x + local_index + wg_id.x + num_wgs.x;
}

struct Input1 {
    @builtin(vertex_index) index: u32,
}

struct Input2 {
    @builtin(instance_index) index: u32,
}

@vertex
fn vertex_two_structs(in1: Input1, in2: Input2) -> @builtin(position) @invariant vec4<f32> {
    var index = 2u;
    return vec4<f32>(f32(in1.index), f32(in2.index), f32(index), 0.0);
}
