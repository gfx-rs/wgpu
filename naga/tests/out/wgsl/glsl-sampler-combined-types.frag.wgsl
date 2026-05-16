struct FragmentOutput {
    @builtin(frag_depth) gl_FragDepth: f32,
}

@group(0) @binding(0) 
var u_sampled: texture_2d<f32>;
@group(0) @binding(1) 
var u_isampled: texture_2d<i32>;
@group(0) @binding(2) 
var u_usampled_3d: texture_3d<u32>;
@group(0) @binding(3) 
var u_cube: texture_cube<f32>;
@group(0) @binding(4) 
var u_array: texture_2d_array<f32>;
@group(1) @binding(0) 
var samp: sampler;
var<private> gl_FragCoord_1: vec4<f32>;
var<private> gl_FragDepth: f32;

fn main_1() {
    var coord2_: vec2<i32>;
    var coord3_: vec3<f32>;
    var c0_: vec4<f32>;
    var c1_: vec4<i32>;
    var c2_: vec4<u32>;
    var c3_: vec4<f32>;
    var c4_: vec4<f32>;

    let _e7 = gl_FragCoord_1;
    coord2_ = vec2<i32>(_e7.xy);
    let _e11 = gl_FragCoord_1;
    let _e12 = _e11.xy;
    coord3_ = vec3<f32>(_e12.x, _e12.y, 0f);
    let _e18 = coord2_;
    let _e20 = textureLoad(u_sampled, _e18, 0i);
    c0_ = _e20;
    let _e22 = coord2_;
    let _e24 = textureLoad(u_isampled, _e22, 0i);
    c1_ = _e24;
    let _e26 = coord2_;
    let _e32 = textureLoad(u_usampled_3d, vec3<i32>(_e26.x, _e26.y, 0i), 0i);
    c2_ = _e32;
    let _e34 = coord3_;
    let _e35 = textureSample(u_cube, samp, _e34);
    c3_ = _e35;
    let _e37 = coord3_;
    let _e41 = textureSample(u_array, samp, _e37.xy, i32(_e37.z));
    c4_ = _e41;
    let _e44 = c0_;
    let _e46 = c1_;
    let _e50 = c2_;
    let _e54 = c3_;
    let _e57 = c4_;
    gl_FragDepth = ((((_e44.x + f32(_e46.x)) + f32(_e50.x)) + _e54.x) + _e57.x);
    return;
}

@fragment 
fn main(@builtin(position) gl_FragCoord: vec4<f32>) -> FragmentOutput {
    gl_FragCoord_1 = gl_FragCoord;
    main_1();
    let _e3 = gl_FragDepth;
    return FragmentOutput(_e3);
}
