@group(0) @binding(0)
var image: texture_storage_2d<r32uint, atomic>;

@compute
@workgroup_size(4, 4, 1)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group_id: vec3<u32>) {
    let pixel = id + group_id * 4;
    textureAtomicMax(image, pixel.xy, u32(pixel.x));

    storageBarrier();

    textureAtomicMin(image, pixel.xy, u32(pixel.y));
}