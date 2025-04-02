const g0 = 1;
const g1 = 1u;
const g2 = 1.0;
const g3 = 1.0f;
const g4 = vec4<i32>();
const g5 = vec4(1i);
const g6 = mat2x2<f32>(vec2(), vec2());
const g7 = mat2x2(vec2(1.0, 1), vec2(1, 1));

@compute @workgroup_size(1)
fn main() {
    // Expose some constants that wouldn't otherwise be in the output
    // because they don't have concrete types.
    var g0x = g0;
    var g2x = g2;
    var g7x = g7;

    const c0 = 1;
    const c1 = 1u;
    const c2 = 1.0;
    const c3 = 1.0f;
    const c4 = vec4<i32>();
    const c5 = vec4(1i);
    const c6 = mat2x2<f32>(vec2(), vec2());
    const c7 = mat2x2(vec2(1.0, 1), vec2(1, 1));

    // Local constants are not emitted in most cases.
    // See logic for `Statement::Emit` in `back::wgsl::Writer::write_stmt`.
    var c0x = c0;
    var c1x = c1;
    var c2x = c2;
    var c3x = c3;
    var c4x = c4;
    var c5x = c5;
    var c6x = c6;
    var c7x = c7;

    let l0 = 1;
    let l1 = 1u;
    let l2 = 1.0;
    let l3 = 1.0f;
    let l4 = vec4<i32>();
    let l5 = vec4(1i);
    let l6 = mat2x2<f32>(vec2(), vec2());
    let l7 = mat2x2(vec2(1.0, 1), vec2(1, 1));

    // Let bindings that evaluate to literals or a `ZeroValue` expression are
    // not emitted. See `ConstantEvaluator::append_expr`. `vec4(1i)` is emitted
    // because it is translated to a `Splat` expression.
    var l0x = l0;
    var l1x = l1;
    var l2x = l2;
    var l3x = l3;
    var l4x = l4;

    var v0 = 1;
    var v1 = 1u;
    var v2 = 1.0;
    var v3 = 1.0f;
    var v4 = vec4<i32>();
    var v5 = vec4(1i);
    var v6 = mat2x2<f32>(vec2(), vec2());
    var v7 = mat2x2(vec2(1.0, 1), vec2(1, 1));
}
