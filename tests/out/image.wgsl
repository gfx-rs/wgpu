[[group(0), binding(1)]]
var image_src: [[access(read)]] texture_storage_2d<rgba8uint>;
[[group(0), binding(2)]]
var image_dst: [[access(write)]] texture_storage_1d<r32uint>;
[[group(0), binding(0)]]
var image_1d: texture_1d<f32>;
[[group(0), binding(1)]]
var image_2d: texture_2d<f32>;
[[group(0), binding(2)]]
var image_2d_array: texture_2d_array<f32>;
[[group(0), binding(3)]]
var image_cube: texture_cube<f32>;
[[group(0), binding(4)]]
var image_cube_array: texture_cube_array<f32>;
[[group(0), binding(5)]]
var image_3d: texture_3d<f32>;
[[group(0), binding(6)]]
var image_aa: texture_multisampled_2d<f32>;

[[stage(compute), workgroup_size(16, 1, 1)]]
fn main([[builtin(local_invocation_id)]] local_id: vec3<u32>) {
    let _e10: vec2<i32> = ((textureDimensions(image_src) * vec2<i32>(local_id.xy)) % vec2<i32>(10, 20));
    let _e11: vec4<u32> = textureLoad(image_src, _e10);
    textureStore(image_dst, _e10[0], _e11);
    return;
}

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    return vec4<f32>(f32(((((((((((((((((((textureDimensions(image_1d) + textureDimensions(image_2d)[1]) + textureDimensions(image_2d, 1)[1]) + textureDimensions(image_2d_array)[1]) + textureDimensions(image_2d_array, 1)[1]) + textureNumLayers(image_2d_array)) + textureDimensions(image_cube)[1]) + textureDimensions(image_cube, 1)[1]) + textureDimensions(image_cube_array)[1]) + textureDimensions(image_cube_array, 1)[1]) + textureNumLayers(image_cube_array)) + textureDimensions(image_3d)[2]) + textureDimensions(image_3d, 1)[2]) + textureNumSamples(image_aa)) + textureNumLevels(image_2d)) + textureNumLevels(image_2d_array)) + textureNumLevels(image_3d)) + textureNumLevels(image_cube)) + textureNumLevels(image_cube_array))));
}
