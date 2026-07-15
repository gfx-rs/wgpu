struct RawTlasInstance {
    transform: array<f32, 12>,        // row-major 3x4 affine (VkTransformMatrixKHR)
    custom_data_and_mask: u32,        // custom_data (low 24 bits) | mask << 24
    sbt_offset_and_flags: u32,        // shader-binding-table offset (low 24) | flags << 24
    blas_address: vec2<u32>,          // BLAS device address, low u32 then high u32
}

// count * 12 floats, one row-major [f32; 12] transform per instance, contiguous.
@group(0) @binding(0) var<storage, read> transforms: array<f32>;
// count u32s: per-instance index into `blas_addresses`.
@group(0) @binding(1) var<storage, read> blas_indices: array<u32>;
// one u64 device address per unique BLAS (from `Blas::handle()`), as (low, high) u32.
@group(0) @binding(2) var<storage, read> blas_addresses: array<vec2<u32>>;
// count u32s: per-instance custom data (only the low 24 bits are used), surfaced to shaders as
// `RayIntersection::instance_custom_data`.
@group(0) @binding(3) var<storage, read> custom_data: array<u32>;
// output: count `RawTlasInstance` records.
@group(0) @binding(4) var<storage, read_write> out_instances: array<RawTlasInstance>;

struct Params {
    count: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if i >= params.count {
        return;
    }

    var inst: RawTlasInstance;
    let base = i * 12u;
    for (var k = 0u; k < 12u; k = k + 1u) {
        inst.transform[k] = transforms[base + k];
    }
    inst.custom_data_and_mask = (custom_data[i] & 0xFFFFFFu) | (0xFFu << 24u);
    inst.sbt_offset_and_flags = 0u;
    inst.blas_address = blas_addresses[blas_indices[i]];

    out_instances[i] = inst;
}
