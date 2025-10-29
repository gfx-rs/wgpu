struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

const VERTICES: array<vec3<f32>, 3> = array<vec3<f32>, 3>(
    vec3<f32>(-0.5, 0.0, 0.0),
    vec3<f32>(0.5, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
);

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4(VERTICES[idx], 1.0);
    return output;
}

@fragment
fn fs_y_main(input: VertexOutput) -> @location(0) f32 {
    let color = vec3<f32>(1.0);
    let conversion_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
    return clamp(dot(color, conversion_weights), 0.0, 1.0);
}

@fragment
fn fs_uv_main(input: VertexOutput) -> @location(0) vec2<f32> {
    let color = vec3<f32>(1.0);
    let conversion_weights = mat3x2<f32>(
        -0.1146, 0.5,
        -0.3854, -0.4542,
        0.5, -0.0458,
    );
    let conversion_bias = vec2<f32>(0.5, 0.5);
    return clamp(conversion_weights * color + conversion_bias, vec2(0.0, 0.0), vec2(1.0, 1.0));
}
