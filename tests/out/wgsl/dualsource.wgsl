struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(0) @second_blend_source mask: vec4<f32>,
}

@fragment 
fn main(@builtin(position) position: vec4<f32>) -> FragmentOutput {
    var color: vec4<f32>;
    var mask: vec4<f32>;

    color = vec4<f32>(0.4, 0.3, 0.2, 0.1);
    mask = vec4<f32>(0.9, 0.8, 0.7, 0.6);
    let _e13 = color;
    let _e14 = mask;
    return FragmentOutput(_e13, _e14);
}
