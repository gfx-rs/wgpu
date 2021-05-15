struct VertexOutput {
    [[builtin(position)]] position: vec4<f32>;
    [[location(1), interpolate(perspective)]] varying: f32;
};

struct FragmentOutput {
    [[builtin(frag_depth)]] depth: f32;
    [[builtin(sample_mask)]] sample_mask: u32;
    [[location(0)]] color: f32;
};

[[stage(vertex)]]
fn vertex([[builtin(vertex_index)]] vertex_index: u32, [[builtin(instance_index)]] instance_index: u32, [[location(10)]] color1: u32) -> VertexOutput {
    return VertexOutput(vec4<f32>(1.0), f32(((vertex_index + instance_index) + color1)));
}

[[stage(fragment)]]
fn fragment(in: VertexOutput, [[builtin(front_facing)]] front_facing: bool, [[builtin(sample_index)]] sample_index: u32, [[builtin(sample_mask)]] sample_mask1: u32) -> FragmentOutput {
    return FragmentOutput(in.varying, (sample_mask1 & (1u << sample_index)), select(0.0, 1.0, front_facing));
}

[[stage(compute), workgroup_size(1, 1, 1)]]
fn compute([[builtin(global_invocation_id)]] global_id: vec3<u32>, [[builtin(local_invocation_id)]] local_id: vec3<u32>, [[builtin(local_invocation_index)]] local_index: u32, [[builtin(workgroup_id)]] wg_id: vec3<u32>) {
    return;
}
