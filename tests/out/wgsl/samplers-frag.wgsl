struct FragmentOutput {
    [[location(0)]] o_color: vec4<f32>;
};

[[group(1), binding(0)]]
var tex: texture_2d_array<f32>;
[[group(1), binding(1)]]
var samp: sampler;
var<private> v_TexCoord1: vec2<f32>;
var<private> o_color: vec4<f32>;

fn main1() {
    let _e4: vec4<f32> = o_color;
    let _e6: vec2<f32> = v_TexCoord1;
    let _e9: vec2<f32> = v_TexCoord1;
    let _e11: vec3<f32> = vec3<f32>(_e9, 0.0);
    let _e15: vec4<f32> = textureSample(tex, samp, _e11.xy, i32(_e11.z));
    o_color.x = _e15.x;
    o_color.y = _e15.y;
    o_color.z = _e15.z;
    o_color.w = _e15.w;
    return;
}

[[stage(fragment)]]
fn main([[location(0)]] v_TexCoord: vec2<f32>) -> FragmentOutput {
    v_TexCoord1 = v_TexCoord;
    main1();
    let _e11: vec4<f32> = o_color;
    return FragmentOutput(_e11);
}
