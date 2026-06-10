struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
    @location(1) o_shadow: f32,
}

@group(0) @binding(0) 
var u_lightmap: texture_2d<f32>;
@group(0) @binding(3) 
var u_lightmap_sampler: sampler;
@group(0) @binding(1) 
var u_shadow: texture_depth_2d;
@group(0) @binding(4) 
var u_shadow_sampler: sampler_comparison;
@group(0) @binding(2) 
var u_overlay: texture_2d<f32>;
@group(0) @binding(5) 
var u_overlay_sampler: sampler;
var<private> v_uv_1: vec2<f32>;
var<private> v_shadow_coord_1: vec3<f32>;
var<private> o_color: vec4<f32>;
var<private> o_shadow: f32;

fn sample_tex(tex: texture_2d<f32>, param: sampler, uv: vec2<f32>) -> vec4<f32> {
    var uv_1: vec2<f32>;

    uv_1 = uv;
    let _e4 = uv_1;
    let _e5 = textureSample(tex, param, _e4);
    return _e5;
}

fn sample_shadow(sm: texture_depth_2d, param_1: sampler_comparison, coord: vec3<f32>) -> f32 {
    var coord_1: vec3<f32>;

    coord_1 = coord;
    let _e4 = coord_1;
    let _e7 = textureSampleCompare(sm, param_1, _e4.xy, _e4.z);
    return _e7;
}

fn inner(tex_1: texture_2d<f32>, param_2: sampler, uv_2: vec2<f32>) -> vec4<f32> {
    var uv_3: vec2<f32>;

    uv_3 = uv_2;
    let _e4 = uv_3;
    let _e5 = textureSample(tex_1, param_2, _e4);
    return _e5;
}

fn outer(tex_2: texture_2d<f32>, param_3: sampler, uv_4: vec2<f32>) -> vec4<f32> {
    var uv_5: vec2<f32>;

    uv_5 = uv_4;
    let _e4 = uv_5;
    let _e5 = inner(tex_2, param_3, _e4);
    return _e5;
}

fn blend(a: texture_2d<f32>, param_4: sampler, b: texture_2d<f32>, param_5: sampler, uv_6: vec2<f32>) -> vec4<f32> {
    var uv_7: vec2<f32>;

    uv_7 = uv_6;
    let _e6 = uv_7;
    let _e7 = textureSample(a, param_4, _e6);
    let _e8 = uv_7;
    let _e9 = textureSample(b, param_5, _e8);
    return mix(_e7, _e9, vec4(0.5f));
}

fn sample_proto(tex_3: texture_2d<f32>, param_6: sampler, uv_8: vec2<f32>) -> vec4<f32> {
    var uv_9: vec2<f32>;

    uv_9 = uv_8;
    let _e4 = uv_9;
    let _e5 = textureSample(tex_3, param_6, _e4);
    return _e5;
}

fn main_1() {
    let _e10 = v_uv_1;
    let _e11 = sample_tex(u_lightmap, u_lightmap_sampler, _e10);
    o_color = _e11;
    let _e12 = v_shadow_coord_1;
    let _e13 = sample_shadow(u_shadow, u_shadow_sampler, _e12);
    o_shadow = _e13;
    let _e14 = o_color;
    let _e15 = v_uv_1;
    let _e16 = outer(u_lightmap, u_lightmap_sampler, _e15);
    o_color = (_e14 + _e16);
    let _e18 = o_color;
    let _e19 = v_uv_1;
    let _e20 = blend(u_lightmap, u_lightmap_sampler, u_overlay, u_overlay_sampler, _e19);
    o_color = (_e18 + _e20);
    let _e22 = o_color;
    let _e23 = v_uv_1;
    let _e24 = sample_proto(u_lightmap, u_lightmap_sampler, _e23);
    o_color = (_e22 + _e24);
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
