// FXC requires that all non-system values get written first, but doesn't consider SV_PrimitiveID a system value.
// This test ensures that the inputs are written in the right order

enable primitive_index;

@fragment
fn func(@location(0) input_location: f32, @builtin(position) arbitrary_position: vec4<f32>, @builtin(primitive_index) index: u32) -> @location(0) vec4<f32> {
    return vec4(arbitrary_position.xy, input_location, f32(index));
}
