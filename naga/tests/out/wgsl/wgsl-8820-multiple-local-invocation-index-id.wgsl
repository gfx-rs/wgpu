struct Input {
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index) local_invocation_index: u32,
}

var<workgroup> wg_var: u32;

@compute @workgroup_size(1, 1, 1) 
fn compute1_(input: Input) {
    wg_var = (input.local_invocation_index * 2u);
    let _e6 = wg_var;
    wg_var = (_e6 + input.local_invocation_id.x);
    return;
}
