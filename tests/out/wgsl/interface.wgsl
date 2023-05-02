struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(1) _varying: f32,
}

struct FragmentOutput {
    @builtin(frag_depth) depth: f32,
    @builtin(sample_mask) sample_mask: u32,
    @location(0) color: f32,
}

struct Input1_ {
    @builtin(vertex_index) index: u32,
}

struct Input2_ {
    @builtin(instance_index) index: u32,
}

var<workgroup> output: array<u32,1>;

@vertex 
fn vertex(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32, @location(10) @interpolate(flat) color: u32) -> VertexOutput {
    let tmp: u32 = ((vertex_index + instance_index) + color);
    return VertexOutput(vec4<f32>(1.0), f32(tmp));
}

@fragment 
fn fragment(in: VertexOutput, @builtin(front_facing) front_facing: bool, @builtin(sample_index) sample_index: u32, @builtin(sample_mask) sample_mask: u32) -> FragmentOutput {
    let mask: u32 = (sample_mask & (1u << sample_index));
    let color_1: f32 = select(0.0, 1.0, front_facing);
    return FragmentOutput(in._varying, mask, color_1);
}

@compute @workgroup_size(1, 1, 1) 
fn compute(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(local_invocation_index) local_index: u32, @builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wgs: vec3<u32>) {
    output[0] = ((((global_id.x + local_id.x) + local_index) + wg_id.x) + num_wgs.x);
    return;
}

@vertex 
fn vertex_two_structs(in1_: Input1_, in2_: Input2_) -> @builtin(position) @invariant vec4<f32> {
    var index: u32;

    index = 2u;
    let _e8: u32 = index;
    return vec4<f32>(f32(in1_.index), f32(in2_.index), f32(_e8), 0.0);
}
