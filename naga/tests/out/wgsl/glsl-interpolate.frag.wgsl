struct FragmentOutput {
    @location(0) o_color: vec4<f32>,
}

var<private> tex_coord_1: vec2<f32>;
var<private> index_1: i32;
var<private> o_color: vec4<f32>;

fn main_1() {
    let _e3 = tex_coord_1;
    let _e5 = index_1;
    o_color = vec4<f32>(_e3.x, _e3.y, 0f, f32(_e5));
    return;
}

@fragment 
fn main(@location(0) tex_coord: vec2<f32>, @location(1) @interpolate(flat) index: i32) -> FragmentOutput {
    tex_coord_1 = tex_coord;
    index_1 = index;
    main_1();
    let _e5 = o_color;
    return FragmentOutput(_e5);
}
