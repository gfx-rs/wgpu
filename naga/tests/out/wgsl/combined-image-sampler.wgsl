var<private> color: vec4<f32>;
@group(0) @binding(1) 
var tex: texture_2d<f32>;
@group(1) @binding(1) 
var _tex_sampler: sampler;

fn main_1() {
    let _e4 = textureSample(tex, _tex_sampler, vec2<f32>(0.0, 0.0));
    color = _e4;
    return;
}

@fragment 
fn main() -> @location(0) vec4<f32> {
    main_1();
    let _e1 = color;
    return _e1;
}
