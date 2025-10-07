var<private> input_u002e_texture_coordinates_1: vec2<f32>;
@group(0) @binding(0) 
var texture: texture_depth_2d;
@group(0) @binding(1) 
var depth_sampler: sampler_comparison;
var<private> entryPointParam_main: vec4<f32>;

fn main_1() {
    let _e5 = input_u002e_texture_coordinates_1;
    let _e6 = textureGatherCompare(texture, depth_sampler, _e5, 0.5f);
    entryPointParam_main = _e6;
    return;
}

@fragment 
fn main(@location(0) input_u002e_texture_coordinates: vec2<f32>) -> @location(0) vec4<f32> {
    input_u002e_texture_coordinates_1 = input_u002e_texture_coordinates;
    main_1();
    let _e3 = entryPointParam_main;
    return _e3;
}
