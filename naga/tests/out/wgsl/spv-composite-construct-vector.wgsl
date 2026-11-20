var<private> uv_1: vec2<f32>;
var<private> colour: vec4<f32>;

fn main_1() {
    let _e5 = uv_1;
    colour = (vec4<f32>(_e5.x, _e5.y, 0f, 1f) + vec4<f32>(_e5.x, _e5.y, vec2<f32>(0f, 1f).x, vec2<f32>(0f, 1f).y));
    return;
}

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = colour;
    return _e3;
}
