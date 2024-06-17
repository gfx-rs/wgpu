@group(0) @binding(0)
var tex: binding_array<texture_2d<f32>>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) instance_index: u32,
}

@vertex
fn vs_main(@builtin(instance_index) instance_index: u32) -> VertexOutput {
    return VertexOutput(
        vec4f(0.0, 0.0, 0.0, 1.0),
        instance_index
    );
}

@fragment
fn fs_main(vs_in: VertexOutput) -> @location(0) vec4f {
    return textureLoad(tex[7 * vs_in.instance_index + 0], vec2u(0), 0) +
           textureLoad(tex[7 * vs_in.instance_index + 1], vec2u(0), 0) +
           textureLoad(tex[7 * vs_in.instance_index + 2], vec2u(0), 0) +
           textureLoad(tex[7 * vs_in.instance_index + 3], vec2u(0), 0) +
           textureLoad(tex[7 * vs_in.instance_index + 4], vec2u(0), 0) +
           textureLoad(tex[7 * vs_in.instance_index + 5], vec2u(0), 0) +
           textureLoad(tex[7 * vs_in.instance_index + 6], vec2u(0), 0); 
}
