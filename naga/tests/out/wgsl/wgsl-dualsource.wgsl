enable dual_source_blending;

struct FragmentOutput {
    @location(0) @blend_src(0) output0_: vec4<f32>,
    @location(0) @blend_src(1) output1_: vec4<f32>,
}

@fragment 
fn main() -> FragmentOutput {
    return FragmentOutput(vec4<f32>(0.4f, 0.3f, 0.2f, 0.1f), vec4<f32>(0.9f, 0.8f, 0.7f, 0.6f));
}
