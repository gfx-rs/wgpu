const g1_: u32 = 1u;
const g3_: f32 = 1f;
const g4_: vec4<i32> = vec4<i32>();
const g5_: vec4<i32> = vec4(1i);
const g6_: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0f, 0f), vec2<f32>(0f, 0f));

@compute @workgroup_size(1, 1, 1) 
fn main() {
    var g0x: i32 = 1i;
    var g2x: f32 = 1f;
    var g7x: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 1f), vec2<f32>(1f, 1f));
    var c0x: i32 = 1i;
    var c1x: u32 = 1u;
    var c2x: f32 = 1f;
    var c3x: f32 = 1f;
    var c4x: vec4<i32> = vec4<i32>();
    var c5x: vec4<i32> = vec4(1i);
    var c6x: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0f, 0f), vec2<f32>(0f, 0f));
    var c7x: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 1f), vec2<f32>(1f, 1f));
    var l0x: i32;
    var l1x: u32;
    var l2x: f32;
    var l3x: f32;
    var l4x: vec4<i32>;
    var v0_: i32 = 1i;
    var v1_: u32 = 1u;
    var v2_: f32 = 1f;
    var v3_: f32 = 1f;
    var v4_: vec4<i32> = vec4<i32>();
    var v5_: vec4<i32> = vec4(1i);
    var v6_: mat2x2<f32> = mat2x2<f32>(vec2<f32>(0f, 0f), vec2<f32>(0f, 0f));
    var v7_: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 1f), vec2<f32>(1f, 1f));

    let l5_ = vec4(1i);
    let l6_ = mat2x2<f32>(vec2<f32>(0f, 0f), vec2<f32>(0f, 0f));
    let l7_ = mat2x2<f32>(vec2<f32>(1f, 1f), vec2<f32>(1f, 1f));
    l0x = 1i;
    l1x = 1u;
    l2x = 1f;
    l3x = 1f;
    l4x = vec4<i32>();
    return;
}
