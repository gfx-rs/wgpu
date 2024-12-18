struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main( @builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    out.uv = vec2<f32>(
        f32((vi << 1u) & 2u),
        f32(vi & 2u),
    );
    out.clip_position = vec4<f32>(out.uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv.y = 1.0 - out.uv.y;
    return out;
}

@group(0)
@binding(0)
var image_texture: texture_2d<f32>;
@group(0)
@binding(1)
var image_sampler: sampler;

struct FragmentOutput {
  @location(0) red_target : vec4<f32>,
  @location(1) green_target : vec4<f32>,
}

@fragment
fn fs_multi_main(vs: VertexOutput) -> FragmentOutput {
    let smp = textureSample(image_texture, image_sampler, vs.uv).x;

    var output: FragmentOutput;
    output.red_target = vec4<f32>(smp, 0.0, 0.0, 1.0);
    output.green_target = vec4<f32>(0.0, smp, 0.0, 1.0);
    return output;
}

@fragment
fn fs_display_main(vs: VertexOutput) -> @location(0) vec4<f32> {
    let smp = textureSample(image_texture, image_sampler, vs.uv).xyz;
    return vec4<f32>(smp, 1.0);
}
