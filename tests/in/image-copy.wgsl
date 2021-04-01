[[group(0), binding(1)]]
var image_src: [[access(read)]] texture_storage_2d<rgba8uint>;
[[group(0), binding(2)]]
var image_dst: [[access(write)]] texture_storage_1d<r32uint>;

[[stage(compute), workgroup_size(16)]]
fn main(
    [[builtin(local_invocation_id)]] local_id: vec3<u32>,
    //TODO: https://github.com/gpuweb/gpuweb/issues/1590
    //[[builtin(workgroup_size)]] wg_size: vec3<u32>
) {
    const dim = textureDimensions(image_src);
    const itc = dim * vec2<i32>(local_id.xy) % vec2<i32>(10, 20);
    const value = textureLoad(image_src, itc);
    textureStore(image_dst, itc.x, value);
}
