struct Block {
    basis: mat2x2<f32>,
    offset: vec2<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

var<private> transforms_6: array<mat3x2<f32>, 2>;
var<private> data: Block;
var<private> color: vec4<f32>;

fn main_1() {
    let _e5 = transforms_6[0][1];
    let _e8 = transforms_6[1][2];
    let _e9 = (_e5 + _e8);
    let _e12 = data.basis[1];
    let _e13 = data;
    let _e15 = (_e12 + _e13.offset);
    color = vec4<f32>(_e9.x, _e9.y, _e15.x, _e15.y);
    return;
}

@fragment
fn main(@location(0) @interpolate(flat) transforms: vec2<f32>, @location(1) @interpolate(flat) transforms_1: vec2<f32>, @location(2) @interpolate(flat) transforms_2: vec2<f32>, @location(3) @interpolate(flat) transforms_3: vec2<f32>, @location(4) @interpolate(flat) transforms_4: vec2<f32>, @location(5) @interpolate(flat) transforms_5: vec2<f32>, @location(6) basis: vec2<f32>, @location(7) basis_1: vec2<f32>, @location(8) offset: vec2<f32>) -> FragmentOutput {
    transforms_6[0][0] = transforms;
    transforms_6[0][1] = transforms_1;
    transforms_6[0][2] = transforms_2;
    transforms_6[1][0] = transforms_3;
    transforms_6[1][1] = transforms_4;
    transforms_6[1][2] = transforms_5;
    data.basis[0] = basis;
    data.basis[1] = basis_1;
    data.offset = offset;
    main_1();
    let _e24 = color;
    return FragmentOutput(_e24);
}
