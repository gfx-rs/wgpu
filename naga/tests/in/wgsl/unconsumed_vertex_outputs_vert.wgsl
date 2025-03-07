// Out of order to test sorting.
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(1) value: f32,
    @location(2) unused_value2: vec4<f32>,
    @location(0) unused_value: f32,
    @location(3) value2: f32,
}

@vertex
fn vs_main() -> VertexOut {
    return VertexOut(vec4(1.0), 1.0, vec4(2.0), 1.0, 0.5);
}
