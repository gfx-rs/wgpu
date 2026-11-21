@group(0) @binding(0)
var image_u: texture_storage_2d_array<r32uint, atomic>;
@group(0) @binding(1)
var image_s: texture_storage_2d_array<r32sint, atomic>;

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    textureAtomicMax(image_u, vec2<i32>(0, 0), 1, 1u);
    textureAtomicMin(image_s, vec2<i32>(0, 0), 1u, 1i);
}
