
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main( @builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate a triangle that covers the whole screen
    out.uv = vec2<f32>(
        f32((vi << 1u) & 2u),
        f32(vi & 2u),
    );
    out.clip_position = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
    // We need to invert the y coordinate so the image
    // is not upside down
    out.uv.y = 1.0 - out.uv.y;
    return out;
}


struct FragmentOutput {
  @location(0) target_a : vec4<f32>,
  @location(1) target_b : vec4<f32>,
}

@group(0)
@binding(0)
var image_texture: texture_2d<f32>;
@group(0)
@binding(1)
var image_sampler: sampler;

@fragment
fn fs_main(vs: VertexOutput) -> FragmentOutput {
    let smp = textureSample(image_texture, image_sampler, vs.uv).x;

    var output: FragmentOutput;
    output.target_a = vec4<f32>(smp, 0.0, 0.0, 1.0);
    output.target_b = vec4<f32>(0.0, smp, 0.0, 1.0);
    return output;
}
