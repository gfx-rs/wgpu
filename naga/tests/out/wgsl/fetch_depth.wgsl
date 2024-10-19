struct type_2 {
    member: f32,
}

struct type_4 {
    member: vec2<u32>,
}

@group(0) @binding(0) 
var<storage, read_write> global: type_2;
@group(0) @binding(1) 
var<storage> global_1: type_4;
@group(0) @binding(2) 
var global_2: texture_depth_2d;

fn function() {
    let _e6 = global_1.member;
    let _e7 = textureLoad(global_2, _e6, 0i);
    global.member = vec4(_e7).x;
    return;
}

@compute @workgroup_size(32, 1, 1) 
fn cullfetch_depth() {
    function();
}
