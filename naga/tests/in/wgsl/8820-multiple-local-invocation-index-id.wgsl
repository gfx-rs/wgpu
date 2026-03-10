var<workgroup> wg_var: u32;

struct Input {
    @builtin(local_invocation_id)
    local_invocation_id: vec3<u32>,
    @builtin(local_invocation_index)
    local_invocation_index: u32,
}

@compute
@workgroup_size(1)
fn compute1(input: Input) {
    wg_var = input.local_invocation_index * 2;
    wg_var += input.local_invocation_id.x;
}
