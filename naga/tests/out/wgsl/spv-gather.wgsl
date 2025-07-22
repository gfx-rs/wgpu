var<private> input_u002e_texture_coordinates_1: vec2<f32>;
@group(0) @binding(0) 
var texture: texture_2d<f32>;
@group(0) @binding(1) 
var linear_sampler: sampler;
var<private> entryPointParam_main: vec4<f32>;

fn main_1() {
    let _e4 = input_u002e_texture_coordinates_1;
    let _e5 = textureGather(1, texture, linear_sampler, _e4);
    entryPointParam_main = _e5;
    return;
}

@fragment 
fn main(@location(0) input_u002e_texture_coordinates: vec2<f32>) -> @location(0) vec4<f32> {
    input_u002e_texture_coordinates_1 = input_u002e_texture_coordinates;
    main_1();
    let _e3 = entryPointParam_main;
    return _e3;
}
