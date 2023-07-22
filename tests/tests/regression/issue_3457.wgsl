struct DoubleVertexIn {
    @location(0) position: vec4<f32>,
    @location(5) value: f32,
}

struct DoubleVertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) value: f32,
}

@vertex
fn double_buffer_vert(v_in: DoubleVertexIn) -> DoubleVertexOut {
    return DoubleVertexOut(v_in.position, v_in.value);
}

@fragment
fn double_buffer_frag(v_out: DoubleVertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(v_out.value);
}

struct SingleVertexIn {
    @location(0) position: vec4<f32>,
}

@vertex
fn single_buffer_vert(v_in: SingleVertexIn) -> @builtin(position) vec4<f32> {
    return v_in.position;
}

@fragment
fn single_buffer_frag() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
