struct Packed {
    member: vec3<f32>,
    member_1: f32,
}

const VEC: vec2<f32> = vec2<f32>(1f, 2f);
const MATRIX: mat2x2<f32> = mat2x2<f32>(VEC, VEC);
const MATRICES: array<mat2x2<f32>, 2> = array<mat2x2<f32>, 2>(MATRIX, MATRIX);
const PACKED: Packed = Packed(vec3<f32>(1f, 1f, 1f), 2f);
const PACKED_ARRAY: array<Packed, 2> = array<Packed, 2>(PACKED, PACKED);

var<private> color: vec4<f32>;

fn main_1() {
    color = vec4<f32>(MATRICES[1][0].y, PACKED_ARRAY[1].member.z, PACKED_ARRAY[0].member_1, 1f);
    return;
}

@fragment
fn main() -> @location(0) vec4<f32> {
    main_1();
    let _e1 = color;
    return _e1;
}
