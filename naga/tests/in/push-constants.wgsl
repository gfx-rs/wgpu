struct PushConstants {
    multiplier: f32
}
var<push_constant> pc: PushConstants;

struct FragmentIn {
    @location(0) color: vec4<f32>
}

@vertex
fn vert_main(
  @location(0) pos : vec2<f32>,
  @builtin(instance_index) ii: u32,
  @builtin(vertex_index) vi: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(ii) * f32(vi) * pc.multiplier * pos, 0.0, 1.0);
}

@fragment
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    return in.color * pc.multiplier;
}
