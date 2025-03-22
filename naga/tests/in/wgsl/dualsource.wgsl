enable dual_source_blending;

struct FragmentOutput {
    @location(0) @blend_src(0) output0: vec4<f32>,
    @location(0) @blend_src(1) output1: vec4<f32>,
}

@fragment
fn main() -> FragmentOutput {
    return FragmentOutput(vec4(0.4,0.3,0.2,0.1), vec4(0.9,0.8,0.7,0.6));
}
