struct Block {
    offset: vec2<f32>,
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
}

var<private> transforms_4: array<mat2x2<f32>, 2>;
var<private> basis_2: mat2x2<f32>;
var<private> data: Block;
var<private> bias_1: vec2<f32>;
var<private> scale_1: f32;
var<private> color: vec4<f32>;

fn main_1() {
    let _e8 = transforms_4[0][0];
    let _e11 = transforms_4[1][1];
    let _e14 = basis_2[0];
    let _e16 = data;
    let _e19 = bias_1;
    let _e21 = scale_1;
    let _e22 = (((((_e8 + _e11) + _e14) + _e16.offset) + _e19) * _e21);
    color = vec4<f32>(_e22.x, _e22.y, 0f, 1f);
    return;
}

@fragment
fn main(@location(0) @interpolate(perspective, centroid) transforms: vec2<f32>, @location(1) @interpolate(perspective, centroid) transforms_1: vec2<f32>, @location(2) @interpolate(perspective, centroid) transforms_2: vec2<f32>, @location(3) @interpolate(perspective, centroid) transforms_3: vec2<f32>, @location(4) @interpolate(perspective, sample) basis: vec2<f32>, @location(5) @interpolate(perspective, sample) basis_1: vec2<f32>, @location(6) offset: vec2<f32>, @location(7) @interpolate(perspective, centroid) bias: vec2<f32>, @location(8) @interpolate(perspective, sample) scale: f32) -> FragmentOutput {
    transforms_4[0][0] = transforms;
    transforms_4[0][1] = transforms_1;
    transforms_4[1][0] = transforms_2;
    transforms_4[1][1] = transforms_3;
    basis_2[0] = basis;
    basis_2[1] = basis_1;
    data.offset = offset;
    bias_1 = bias;
    scale_1 = scale;
    main_1();
    let _e24 = color;
    return FragmentOutput(_e24);
}
