struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    // 0, 0
    // 2, 0
    // 0, 2
    // 2, 2
    let v_data = vec2<f32>(f32((vertexIndex << 1u) & 2u), f32(vertexIndex & 2u));
    output.pos = vec4<f32>(v_data - 1.0, 0.0, 1.0);
    output.uv = v_data / 2.0;
    return output;
}

@group(0) @binding(0) var s: sampler;
@group(0) @binding(1) var tex_y: texture_2d<f32>;
@group(0) @binding(2) var tex_uv: texture_2d<f32>;

@fragment
fn fs_main(v_output: VertexOutput) -> @location(0) vec4<f32> {
    let luminance = textureSample(tex_y, s, v_output.uv).r;
    let chrominance = textureSample(tex_uv, s, v_output.uv).rg;
    let rgb = mat3x3<f32>(
        1.000000, 1.000000, 1.000000,
        0.000000,-0.187324, 1.855600,
        1.574800,-0.468124, 0.000000,
    ) * vec3<f32>(luminance, chrominance.r - 0.5, chrominance.g - 0.5);
    return vec4<f32>(rgb, 1.0);
}
