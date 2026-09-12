// Workgroup variable with a struct type must NOT trigger layout decorations.
// The Vulkan spec (VUID-StandaloneSpirv-None-10684) forbids ArrayStride, Offset,
// MatrixStride, and ColMajor on types used by Workgroup storage class variables.
struct Shared {
    data: array<vec4<f32>, 64>,
    count: u32,
}

var<workgroup> wg: Shared;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32) {
    wg.data[lid] = vec4<f32>(f32(lid));
    workgroupBarrier();
    if lid == 0u {
        wg.count = 64u;
    }
}
