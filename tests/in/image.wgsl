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
    let dim = textureDimensions(image_src);
    let itc = dim * vec2<i32>(local_id.xy) % vec2<i32>(10, 20);
    let value = textureLoad(image_src, itc);
    textureStore(image_dst, itc.x, value);
}

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

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    let dim_1d = textureDimensions(image_1d);
    let dim_2d = textureDimensions(image_2d);
    let num_levels_2d = textureNumLevels(image_2d);
    let dim_2d_lod = textureDimensions(image_2d, 1);
    let dim_2d_array = textureDimensions(image_2d_array);
    let num_levels_2d_array = textureNumLevels(image_2d_array);
    let dim_2d_array_lod = textureDimensions(image_2d_array, 1);
    let num_layers_2d = textureNumLayers(image_2d_array);
    let dim_cube = textureDimensions(image_cube);
    let num_levels_cube = textureNumLevels(image_cube);
    let dim_cube_lod = textureDimensions(image_cube, 1);
    let dim_cube_array = textureDimensions(image_cube_array);
    let num_levels_cube_array = textureNumLevels(image_cube_array);
    let dim_cube_array_lod = textureDimensions(image_cube_array, 1);
    let num_layers_cube = textureNumLayers(image_cube_array);
    let dim_3d = textureDimensions(image_3d);
    let num_levels_3d = textureNumLevels(image_3d);
    let dim_3d_lod = textureDimensions(image_3d, 1);
    let num_samples_aa = textureNumSamples(image_aa);

    let sum = dim_1d + dim_2d.y + dim_2d_lod.y + dim_2d_array.y + dim_2d_array_lod.y +
        num_layers_2d + dim_cube.y + dim_cube_lod.y + dim_cube_array.y + dim_cube_array_lod.y +
        num_layers_cube + dim_3d.z + dim_3d_lod.z + num_samples_aa +
        num_levels_2d + num_levels_2d_array + num_levels_3d + num_levels_cube + num_levels_cube_array;
    return vec4<f32>(f32(sum));
}
