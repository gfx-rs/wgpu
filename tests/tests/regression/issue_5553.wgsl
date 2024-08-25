struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) unused_value: f32,
    @location(1) value: f32,
}

struct FragmentIn {
    @builtin(position) position: vec4<f32>,
    // @location(0) unused_value: f32,
    @location(1) value: f32,
}

@vertex
fn vs_main() -> VertexOut {
    return VertexOut(vec4(1.0), 1.0, 1.0);
}

@fragment
fn fs_main(v_out: FragmentIn) -> @location(0) vec4<f32> {
    return vec4<f32>(v_out.value);
}


