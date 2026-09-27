// Workgroup variables must not get ArrayStride/Offset/MatrixStride decorations
// (VUID-StandaloneSpirv-None-10684).
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
