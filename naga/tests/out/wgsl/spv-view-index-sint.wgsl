struct gl_PerVertex {
    @builtin(position) gl_Position: vec4<f32>,
    gl_PointSize: f32,
    gl_ClipDistance: array<f32, 1>,
    gl_CullDistance: array<f32, 1>,
}

var<private> unnamed: gl_PerVertex = gl_PerVertex(vec4<f32>(0f, 0f, 0f, 1f), 1f, array<f32, 1>(), array<f32, 1>());
var<private> gl_ViewIndex_1: i32;

fn main_1() {
    let _e4 = gl_ViewIndex_1;
    unnamed.gl_Position = vec4<f32>(f32(_e4), 0f, 0f, 1f);
    return;
}

@vertex
fn main(@builtin(view_index) gl_ViewIndex: u32) -> @builtin(position) vec4<f32> {
    gl_ViewIndex_1 = i32(gl_ViewIndex);
    main_1();
    let _e5 = unnamed.gl_Position;
    return _e5;
}
