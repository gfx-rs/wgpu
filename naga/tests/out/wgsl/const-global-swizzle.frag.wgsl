struct FragmentOutput {
    @location(0) o_Target: vec4<f32>,
}

const blank: vec2<f32> = vec2<f32>(0f, 1f);

var<private> v_Uv_1: vec2<f32>;
var<private> o_Target: vec4<f32>;

fn main_1() {
    var col: vec2<f32>;

    let _e3 = v_Uv_1;
    col = (_e3.xy * blank);
    let _e7 = col;
    o_Target = vec4<f32>(_e7.x, _e7.y, 0f, 1f);
    return;
}

@fragment 
fn main(@location(0) v_Uv: vec2<f32>) -> FragmentOutput {
    v_Uv_1 = v_Uv;
    main_1();
    let _e9 = o_Target;
    return FragmentOutput(_e9);
}
