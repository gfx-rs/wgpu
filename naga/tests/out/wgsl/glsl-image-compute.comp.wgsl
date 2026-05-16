@group(0) @binding(0) 
var src: texture_storage_2d<rgba8unorm,read>;
@group(0) @binding(1) 
var dst: texture_storage_2d<rgba8unorm,write>;
@group(0) @binding(2) 
var rw: texture_storage_2d<rgba8unorm,read_write>;
var<private> gl_GlobalInvocationID_1: vec3<u32>;

fn main_1() {
    var pos: vec2<i32>;
    var val: vec4<f32>;
    var prev: vec4<f32>;

    let _e4 = gl_GlobalInvocationID_1;
    pos = vec2<i32>(_e4.xy);
    let _e8 = pos;
    let _e9 = textureLoad(src, _e8);
    val = _e9;
    let _e11 = pos;
    let _e12 = val;
    textureStore(dst, _e11, _e12);
    let _e13 = pos;
    let _e14 = textureLoad(rw, _e13);
    prev = _e14;
    let _e16 = pos;
    let _e17 = prev;
    textureStore(rw, _e16, (_e17 * 0.5f));
    return;
}

@compute @workgroup_size(8, 8, 1) 
fn main(@builtin(global_invocation_id) gl_GlobalInvocationID: vec3<u32>) {
    gl_GlobalInvocationID_1 = gl_GlobalInvocationID;
    main_1();
    return;
}
