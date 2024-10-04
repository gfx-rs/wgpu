@group(0) @binding(0)
var image: texture_storage_2d<r64uint, read_write>;

@compute
@workgroup_size(4, 2, 4)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    let data = u64((id.x << 16) | (id.y << 8) | id.z);
    imageAtomicMax(image, id.xy, (u64(100 - id.z) << 32) | data);

    workgroupBarrier();

    imageAtomicMin(image, id.xy, (u64(10 - id.z) << 32) | data);
}