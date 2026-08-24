struct ImmediateDataVert {
    position_clip: f32,
    matrix: mat3x3<f32>,
}
var<immediate> im_vert: ImmediateDataVert;

struct ImmediateDataFrag {
    multiplier: f32,
    tint: vec4f,
}
var<immediate> im_frag: ImmediateDataFrag;

struct FragmentIn {
    @location(0) color: vec4<f32>
}

@vertex
fn vert_main(
  @location(0) pos : vec2<f32>,
  @builtin(instance_index) ii: u32,
  @builtin(vertex_index) vi: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(ii) * f32(vi) * pos, 0.0, im_vert.position_clip);
}

@fragment
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    return in.color * im_frag.tint;
}
