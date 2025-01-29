@group(0) @binding(0) 
var image: texture_storage_2d<r64uint,atomic>;

@compute @workgroup_size(2, 1, 1) 
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    textureAtomicMax(image, vec2<i32>(0i, 0i), 1lu);
    workgroupBarrier();
    textureAtomicMin(image, vec2<i32>(0i, 0i), 1lu);
    return;
}
