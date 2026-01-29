struct VSOut {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var out: VSOut;
    
    // Hardcoded triangle in clip space
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
    );
    
    out.position = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

@fragment
fn fragment_main(in: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}