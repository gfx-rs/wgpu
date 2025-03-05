@group(0) @binding(0)
var image_u: texture_storage_2d<r32uint, atomic>;
@group(0) @binding(1)
var image_s: texture_storage_2d<r32sint, atomic>;

@compute
@workgroup_size(2)
fn cs_main(@builtin(local_invocation_id) id: vec3<u32>) {
    textureAtomicMax(image_u, vec2<i32>(0, 0), 1u);
    textureAtomicMin(image_u, vec2<i32>(0, 0), 1u);
    textureAtomicAdd(image_u, vec2<i32>(0, 0), 1u);
    textureAtomicAnd(image_u, vec2<i32>(0, 0), 1u);
    textureAtomicOr(image_u, vec2<i32>(0, 0), 1u);
    textureAtomicXor(image_u, vec2<i32>(0, 0), 1u);

    textureAtomicMax(image_s, vec2<i32>(0, 0), 1i);
    textureAtomicMin(image_s, vec2<i32>(0, 0), 1i);
    textureAtomicAdd(image_s, vec2<i32>(0, 0), 1i);
    textureAtomicAnd(image_s, vec2<i32>(0, 0), 1i);
    textureAtomicOr(image_s, vec2<i32>(0, 0), 1i);
    textureAtomicXor(image_s, vec2<i32>(0, 0), 1i);
}
