enable wgpu_binding_array;

var<private> input_u002e_texture_coordinates_1: vec2<f32>;
var<private> input_u002e_texture_index_1: u32;
@group(0) @binding(0) 
var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(1) 
var linear_sampler: sampler;
var<private> entryPointParam_main: vec4<f32>;

fn main_1() {
    let _e5 = input_u002e_texture_coordinates_1;
    let _e6 = input_u002e_texture_index_1;
    let _e8 = textureSample(textures[_e6], linear_sampler, _e5);
    entryPointParam_main = _e8;
    return;
}

@fragment 
fn main(@location(0) input_u002e_texture_coordinates: vec2<f32>, @location(1) @interpolate(flat) input_u002e_texture_index: u32) -> @location(0) vec4<f32> {
    input_u002e_texture_coordinates_1 = input_u002e_texture_coordinates;
    input_u002e_texture_index_1 = input_u002e_texture_index;
    main_1();
    let _e5 = entryPointParam_main;
    return _e5;
}
