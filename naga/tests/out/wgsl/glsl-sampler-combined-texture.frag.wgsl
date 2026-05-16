struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
    @location(1) o_shadow: f32,
}

@group(0) @binding(0) 
var u_tex: texture_2d<f32>;
@group(0) @binding(2) 
var u_tex_sampler: sampler;
@group(0) @binding(1) 
var u_shadow: texture_depth_2d;
@group(0) @binding(3) 
var u_shadow_sampler: sampler_comparison;
var<private> v_uv_1: vec2<f32>;
var<private> v_shadow_coord_1: vec3<f32>;
var<private> o_color: vec4<f32>;
var<private> o_shadow: f32;

fn main_1() {
    let _e8 = v_uv_1;
    let _e9 = textureSample(u_tex, u_tex_sampler, _e8);
    o_color = _e9;
    let _e10 = v_shadow_coord_1;
    let _e13 = textureSampleCompare(u_shadow, u_shadow_sampler, _e10.xy, _e10.z);
    o_shadow = _e13;
    return;
}

@fragment 
fn main(@location(0) v_uv: vec2<f32>, @location(1) v_shadow_coord: vec3<f32>) -> FragmentOutput {
    v_uv_1 = v_uv;
    v_shadow_coord_1 = v_shadow_coord;
    main_1();
    let _e5 = o_color;
    let _e7 = o_shadow;
    return FragmentOutput(_e5, _e7);
}
