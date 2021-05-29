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
[[group(1), binding(0)]]
var sampler_cmp: sampler_comparison;
[[group(1), binding(1)]]
var image_2d_depth: texture_depth_2d;

[[stage(compute), workgroup_size(16, 1, 1)]]
fn main([[builtin(local_invocation_id)]] local_id: vec3<u32>) {
    let _e3: vec2<i32> = textureDimensions(image_src);
    let _e10: vec2<i32> = ((_e3 * vec2<i32>(local_id.xy)) % vec2<i32>(10, 20));
    let _e11: vec4<u32> = textureLoad(image_src, _e10);
    textureStore(image_dst, _e10.x, _e11);
    return;
}

[[stage(vertex)]]
fn queries() -> [[builtin(position)]] vec4<f32> {
    let _e9: i32 = textureDimensions(image_1d);
    let _e10: vec2<i32> = textureDimensions(image_2d);
    let _e11: i32 = textureNumLevels(image_2d);
    let _e13: vec2<i32> = textureDimensions(image_2d, 1);
    let _e14: vec2<i32> = textureDimensions(image_2d_array);
    let _e15: i32 = textureNumLevels(image_2d_array);
    let _e17: vec2<i32> = textureDimensions(image_2d_array, 1);
    let _e18: i32 = textureNumLayers(image_2d_array);
    let _e19: vec3<i32> = textureDimensions(image_cube);
    let _e20: i32 = textureNumLevels(image_cube);
    let _e22: vec3<i32> = textureDimensions(image_cube, 1);
    let _e23: vec3<i32> = textureDimensions(image_cube_array);
    let _e24: i32 = textureNumLevels(image_cube_array);
    let _e26: vec3<i32> = textureDimensions(image_cube_array, 1);
    let _e27: i32 = textureNumLayers(image_cube_array);
    let _e28: vec3<i32> = textureDimensions(image_3d);
    let _e29: i32 = textureNumLevels(image_3d);
    let _e31: vec3<i32> = textureDimensions(image_3d, 1);
    let _e32: i32 = textureNumSamples(image_aa);
    return vec4<f32>(f32(((((((((((((((((((_e9 + _e10.y) + _e13.y) + _e14.y) + _e17.y) + _e18) + _e19.y) + _e22.y) + _e23.y) + _e26.y) + _e27) + _e28.z) + _e31.z) + _e32) + _e11) + _e15) + _e29) + _e20) + _e24)));
}

[[stage(fragment)]]
fn sample_comparison() -> [[location(0)]] f32 {
    let _e12: vec2<f32> = vec2<f32>(0.5);
    let _e14: f32 = textureSampleCompare(image_2d_depth, sampler_cmp, _e12, 0.5);
    let _e15: f32 = textureSampleCompareLevel(image_2d_depth, sampler_cmp, _e12, 0.5);
    return (_e14 + _e15);
}
