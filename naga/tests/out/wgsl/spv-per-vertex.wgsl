var<private> global: array<f32, 3>;
var<private> global_1: vec4<f32>;

fn function_() {
    let _e3 = global;
    global_1 = vec4<f32>(_e3[0], _e3[1], _e3[2], 1f);
    return;
}

@fragment 
fn fs_main(@location(0) @interpolate(per_vertex) param: array<f32, 3>) -> @location(0) vec4<f32> {
    global = param;
    function_();
    let _e3 = global_1;
    return _e3;
}
