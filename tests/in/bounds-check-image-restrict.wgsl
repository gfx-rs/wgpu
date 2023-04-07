@group(0) @binding(0)
var image_1d: texture_1d<f32>;

fn test_textureLoad_1d(coords: i32, level: i32) -> vec4<f32> {
   return textureLoad(image_1d, coords, level);
}

@group(0) @binding(1)
var image_2d: texture_2d<f32>;

fn test_textureLoad_2d(coords: vec2<i32>, level: i32) -> vec4<f32> {
   return textureLoad(image_2d, coords, level);
}

@group(0) @binding(2)
var image_2d_array: texture_2d_array<f32>;

fn test_textureLoad_2d_array_u(coords: vec2<i32>, index: u32, level: i32) -> vec4<f32> {
   return textureLoad(image_2d_array, coords, index, level);
}

fn test_textureLoad_2d_array_s(coords: vec2<i32>, index: i32, level: i32) -> vec4<f32> {
   return textureLoad(image_2d_array, coords, index, level);
}

@group(0) @binding(3)
var image_3d: texture_3d<f32>;

fn test_textureLoad_3d(coords: vec3<i32>, level: i32) -> vec4<f32> {
   return textureLoad(image_3d, coords, level);
}

@group(0) @binding(4)
var image_multisampled_2d: texture_multisampled_2d<f32>;

fn test_textureLoad_multisampled_2d(coords: vec2<i32>, _sample: i32) -> vec4<f32> {
   return textureLoad(image_multisampled_2d, coords, _sample);
}

@group(0) @binding(5)
var image_depth_2d: texture_depth_2d;

fn test_textureLoad_depth_2d(coords: vec2<i32>, level: i32) -> f32 {
   return textureLoad(image_depth_2d, coords, level);
}

@group(0) @binding(6)
var image_depth_2d_array: texture_depth_2d_array;

fn test_textureLoad_depth_2d_array_u(coords: vec2<i32>, index: u32, level: i32) -> f32 {
   return textureLoad(image_depth_2d_array, coords, index, level);
}

fn test_textureLoad_depth_2d_array_s(coords: vec2<i32>, index: i32, level: i32) -> f32 {
   return textureLoad(image_depth_2d_array, coords, index, level);
}

@group(0) @binding(7)
var image_depth_multisampled_2d: texture_depth_multisampled_2d;

fn test_textureLoad_depth_multisampled_2d(coords: vec2<i32>, _sample: i32) -> f32 {
   return textureLoad(image_depth_multisampled_2d, coords, _sample);
}

@group(0) @binding(8)
var image_storage_1d: texture_storage_1d<rgba8unorm, write>;

fn test_textureStore_1d(coords: i32, value: vec4<f32>) {
    textureStore(image_storage_1d, coords, value);
}

@group(0) @binding(9)
var image_storage_2d: texture_storage_2d<rgba8unorm, write>;

fn test_textureStore_2d(coords: vec2<i32>, value: vec4<f32>) {
    textureStore(image_storage_2d, coords, value);
}

@group(0) @binding(10)
var image_storage_2d_array: texture_storage_2d_array<rgba8unorm, write>;

fn test_textureStore_2d_array_u(coords: vec2<i32>, array_index: u32, value: vec4<f32>) {
 textureStore(image_storage_2d_array, coords, array_index, value);
}

fn test_textureStore_2d_array_s(coords: vec2<i32>, array_index: i32, value: vec4<f32>) {
 textureStore(image_storage_2d_array, coords, array_index, value);
}

@group(0) @binding(11)
var image_storage_3d: texture_storage_3d<rgba8unorm, write>;

fn test_textureStore_3d(coords: vec3<i32>, value: vec4<f32>) {
    textureStore(image_storage_3d, coords, value);
}

// GLSL output requires that we identify an entry point, so
// that it can tell what "in" and "out" globals to write.
@fragment
fn fragment_shader() -> @location(0) vec4<f32> {
    test_textureLoad_1d(0, 0);
    test_textureLoad_2d(vec2<i32>(), 0);
    test_textureLoad_2d_array_u(vec2<i32>(), 0u, 0);
    test_textureLoad_2d_array_s(vec2<i32>(), 0, 0);
    test_textureLoad_3d(vec3<i32>(), 0);
    test_textureLoad_multisampled_2d(vec2<i32>(), 0);
    // Not yet implemented for GLSL:
    // test_textureLoad_depth_2d(vec2<i32>(), 0);
    // test_textureLoad_depth_2d_array_u(vec2<i32>(), 0u, 0);
    // test_textureLoad_depth_2d_array_s(vec2<i32>(), 0, 0);
    // test_textureLoad_depth_multisampled_2d(vec2<i32>(), 0);
    test_textureStore_1d(0, vec4<f32>());
    test_textureStore_2d(vec2<i32>(), vec4<f32>());
    test_textureStore_2d_array_u(vec2<i32>(), 0u, vec4<f32>());
    test_textureStore_2d_array_s(vec2<i32>(), 0, vec4<f32>());
    test_textureStore_3d(vec3<i32>(), vec4<f32>());

    return vec4<f32>(0.,0.,0.,0.);
}
