struct type_4 {
    @builtin(position) member: vec4<f32>,
    member_1: f32,
    member_2: array<f32, 1>,
    member_3: array<f32, 1>,
}

var<private> global: type_4 = type_4(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> global_1: i32;

fn function() {
    let _e9 = global_1;
    global.member = vec4<f32>(select(1f, -4f, (_e9 == 0i)), select(-1f, 4f, (_e9 == 2i)), 0f, 1f);
    return;
}

@vertex 
fn main(@builtin(vertex_index) param: u32) -> @builtin(position) vec4<f32> {
    global_1 = i32(param);
    function();
    let _e6 = global.member.y;
    global.member.y = -(_e6);
    let _e8 = global.member;
    return _e8;
}
