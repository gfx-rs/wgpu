@group(0) @binding(0)
var image: texture_storage_2d<r64uint, read_write>;

@compute
@workgroup_size(4, 4, 4)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    let l = u64(textureLoad(image, id.xy).r);

    imageAtomicMin(image, id.xy, u64(id.z) + l);

    workgroupBarrier();

    imageAtomicMax(image, id.xy, u64(id.z) + l);
}