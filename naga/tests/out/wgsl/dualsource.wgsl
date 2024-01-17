struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(0) @second_blend_source mask: vec4<f32>,
}

@fragment 
fn main(@builtin(position) position: vec4<f32>) -> FragmentOutput {
    var color: vec4<f32> = vec4<f32>(0.4f, 0.3f, 0.2f, 0.1f);
    var mask: vec4<f32> = vec4<f32>(0.9f, 0.8f, 0.7f, 0.6f);

    let _e13 = color;
    let _e14 = mask;
    return FragmentOutput(_e13, _e14);
}
