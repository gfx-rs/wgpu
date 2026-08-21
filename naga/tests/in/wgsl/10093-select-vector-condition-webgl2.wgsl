// Companion to 10093-select-vector-condition.wgsl, pinned to GLSL ES 3.00
// with WebGL2 semantics: the platform from the bug report, and the version
// furthest below the integer-mix threshold.
//
// GLSL ES 3.00 has no compute shaders, so this uses a fragment entry point
// instead. See https://github.com/gfx-rs/wgpu/issues/10093

struct Inputs {
    @location(0) @interpolate(flat) ia: vec4<i32>,
    @location(1) @interpolate(flat) ib: vec4<i32>,
    @location(2) @interpolate(flat) ua: vec4<u32>,
    @location(3) @interpolate(flat) ub: vec4<u32>,
    @location(4) fa: vec4<f32>,
    @location(5) fb: vec4<f32>,
}

struct Outputs {
    @location(0) out_i: vec4<i32>,
    @location(1) out_u: vec4<u32>,
    @location(2) out_f: vec4<f32>,
    @location(3) out_b: vec4<i32>,
}

@fragment
fn main(in: Inputs) -> Outputs {
    var out: Outputs;

    let icond = in.ia < in.ib;

    // Operands are computed expressions on purpose: without baking they would
    // be re-emitted once per component.
    out.out_i = select(in.ia * 2, in.ib + 7, icond);

    out.out_u = select(in.ua, in.ub, in.ua < in.ub);

    // Float stays on `mix` at every version.
    out.out_f = select(in.fa, in.fb, in.fa < in.fb);

    // vec4<bool> operands as well as a vec4<bool> condition.
    let picked = select(icond, in.ua < in.ub, icond);
    out.out_b = select(vec4<i32>(0), vec4<i32>(1), picked);

    return out;
}
