// Cases from bounds-check-image-restrict that GLSL does not yet support.

@group(0) @binding(0)
var image_depth_2d: texture_depth_2d;

fn test_textureLoad_depth_2d(coords: vec2<i32>, level: i32) -> f32 {
   return textureLoad(image_depth_2d, coords, level);
}

@group(0) @binding(1)
var image_depth_2d_array: texture_depth_2d_array;

fn test_textureLoad_depth_2d_array_u(coords: vec2<i32>, index: u32, level: i32) -> f32 {
   return textureLoad(image_depth_2d_array, coords, index, level);
}

fn test_textureLoad_depth_2d_array_s(coords: vec2<i32>, index: i32, level: i32) -> f32 {
   return textureLoad(image_depth_2d_array, coords, index, level);
}

@group(0) @binding(2)
var image_depth_multisampled_2d: texture_depth_multisampled_2d;

fn test_textureLoad_depth_multisampled_2d(coords: vec2<i32>, _sample: i32) -> f32 {
   return textureLoad(image_depth_multisampled_2d, coords, _sample);
}

@fragment
fn fragment_shader() -> @location(0) vec4<f32> {
    test_textureLoad_depth_2d(vec2<i32>(), 0);
    test_textureLoad_depth_2d_array_u(vec2<i32>(), 0u, 0);
    test_textureLoad_depth_2d_array_s(vec2<i32>(), 0, 0);
    test_textureLoad_depth_multisampled_2d(vec2<i32>(), 0);

    return vec4<f32>(0.,0.,0.,0.);
}
