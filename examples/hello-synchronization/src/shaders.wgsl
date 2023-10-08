@group(0)
@binding(0)
var<storage, read_write> output: array<u32>;

var<workgroup> count: atomic<u32>;

@compute
@workgroup_size(16)
fn patient_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    atomicAdd(&count, 1u);
    workgroupBarrier();
    if (local_id.x == 0u) {
        output[workgroup_id.x] = atomicLoad(&count);
    }
}

@compute
@workgroup_size(16)
fn hasty_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    atomicAdd(&count, 1u);
    if (local_id.x == 0u) {
        output[workgroup_id.x] = atomicLoad(&count);
    }
}