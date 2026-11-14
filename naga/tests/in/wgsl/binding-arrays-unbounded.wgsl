// Unsized binding array with no binding_map entry, so the HLSL backend gets no
// binding_array_size override and must emit an unbounded `[]` array (#10267).
enable wgpu_binding_array;

@group(0) @binding(0)
var textures: binding_array<texture_2d<f32>>;

@fragment
fn main(@location(0) @interpolate(flat) index: u32) -> @location(0) vec4<f32> {
    let dim = textureDimensions(textures[index]);
    return vec4<f32>(f32(dim.x), f32(dim.y), 0.0, 1.0);
}
