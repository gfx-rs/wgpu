var<private> color: vec4<f32>;
@group(0) @binding(1) 
var tex: binding_array<texture_2d<f32>, 2>;
@group(1) @binding(1) 
var _tex_sampler: binding_array<sampler, 2>;

fn main_1() {
    let _e8 = textureSample(tex[0], _tex_sampler[0], vec2<f32>(0.0, 0.0));
    color = _e8;
    let _e9 = color;
    let _e13 = textureSample(tex[1], _tex_sampler[1], vec2<f32>(0.0, 0.0));
    color = (_e9 * _e13);
    return;
}

@fragment 
fn main() -> @location(0) vec4<f32> {
    main_1();
    let _e1 = color;
    return _e1;
}
