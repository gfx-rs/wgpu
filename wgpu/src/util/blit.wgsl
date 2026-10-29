struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;

    out.tex_coords = vec2<f32>(
        f32((vi << 1u) & 2u),
        f32(vi & 2u),
    );

    out.position = vec4<f32>(out.tex_coords * 2.0 - 1.0, 0.0, 1.0);

    // Invert y so the texture is not upside down
    out.tex_coords.y = 1.0 - out.tex_coords.y;
    return out;
}

@group(0) @binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

@fragment
fn fs_main(vs: VertexOutput) -> @location(0) vec4<f32> {
    // Note: we want to copy the texture with the user's provided filter mode,
    // similar to vkCmdBlitImage. We don't want anything to do with mipmapping
    // or partial derivatives, so use `textureSampleLevel` to read from the
    // base texture with the mag filter
    return textureSampleLevel(texture, texture_sampler, vs.tex_coords, 0.0);
}