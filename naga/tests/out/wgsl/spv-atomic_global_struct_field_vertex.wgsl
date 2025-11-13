struct type_5 {
    member: u32,
    member_1: vec2<f32>,
    member_2: atomic<u32>,
}

struct type_6 {
    member: type_5,
}

@group(0) @binding(0) 
var<storage, read_write> global: type_6;
var<private> global_1: vec4<f32> = vec4<f32>(0f, 0f, 0f, 1f);

fn function() {
    let _e7 = global.member.member;
    let _e8 = atomicAdd((&global.member.member_2), _e7);
    let _e9 = f32(_e8);
    let _e12 = global.member.member_1;
    global_1 = vec4<f32>((_e9 * _e12.x), (_e9 * _e12.y), 0f, _e9);
    return;
}

@vertex 
fn global_field_vertex() -> @builtin(position) vec4<f32> {
    function();
    let _e1 = global_1;
    return _e1;
}
