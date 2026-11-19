// Unsigned levels and sample indices, which GLSL image functions take as `int`.

@group(0) @binding(0)
var image_2d_array_u: texture_2d_array<u32>;

fn test_textureLoad_2d_array_u32_level(coords: vec2<i32>, index: i32, level: u32) -> vec4<u32> {
    return textureLoad(image_2d_array_u, coords, index, level);
}

@group(0) @binding(1)
var image_3d: texture_3d<f32>;

fn test_textureLoad_3d_u32_level(coords: vec3<i32>, level: u32) -> vec4<f32> {
    return textureLoad(image_3d, coords, level);
}

@group(0) @binding(2)
var image_multisampled_2d: texture_multisampled_2d<f32>;

fn test_textureLoad_multisampled_2d_u32_sample(coords: vec2<i32>, _sample: u32) -> vec4<f32> {
    return textureLoad(image_multisampled_2d, coords, _sample);
}

// GLSL output requires that we identify an entry point, so
// that it can tell what "in" and "out" globals to write.
@fragment
fn fragment_shader() -> @location(0) vec4<f32> {
    test_textureLoad_2d_array_u32_level(vec2<i32>(), 0, 0u);
    test_textureLoad_3d_u32_level(vec3<i32>(), 0u);
    test_textureLoad_multisampled_2d_u32_sample(vec2<i32>(), 0u);

    return vec4<f32>(0.,0.,0.,0.);
}
