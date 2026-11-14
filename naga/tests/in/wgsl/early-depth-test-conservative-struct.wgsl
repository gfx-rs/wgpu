struct Output {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
@early_depth_test(less_equal)
fn agree_first(@builtin(position) pos: vec4<f32>) -> Output {
    return Output(vec4<f32>(0.4, 0.3, 0.2, 0.1), pos.z - 0.1);
}

@fragment
@early_depth_test(less_equal)
fn agree_second(@builtin(position) pos: vec4<f32>) -> Output {
    return Output(vec4<f32>(0.1, 0.2, 0.3, 0.4), pos.z - 0.2);
}

struct DisagreeingOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
@early_depth_test(greater_equal)
fn disagree_conservative(@builtin(position) pos: vec4<f32>) -> DisagreeingOutput {
    return DisagreeingOutput(vec4<f32>(0.4, 0.3, 0.2, 0.1), pos.z + 0.1);
}

@fragment
fn disagree_unrestricted(@builtin(position) pos: vec4<f32>) -> DisagreeingOutput {
    return DisagreeingOutput(vec4<f32>(0.1, 0.2, 0.3, 0.4), pos.z * 2.0);
}
