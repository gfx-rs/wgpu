// Companion to 10093-select-vector-condition.wgsl, pinned to desktop GLSL 4.40.
//
// `supports_integer_mix()` is `Desktop(v) >= 450 || Embedded(v) >= 310`: one
// `>=` check per version family, joined by `||`. The ES tests above only
// exercise the Embedded arm; this file pins the Desktop family below its
// 4.50 threshold so the Desktop arm gets writer-level coverage too. 440 is
// the highest core version below 450 that still supports the storage
// buffers this compute shader uses. See https://github.com/gfx-rs/wgpu/issues/10093

struct Inputs {
    ia: vec4<i32>,
    ib: vec4<i32>,
    ua: vec4<u32>,
    ub: vec4<u32>,
    fa: vec4<f32>,
    fb: vec4<f32>,
}

@group(0) @binding(0) var<uniform> inputs: Inputs;
@group(0) @binding(1) var<storage, read_write> out_i: vec4<i32>;
@group(0) @binding(2) var<storage, read_write> out_u: vec4<u32>;
@group(0) @binding(3) var<storage, read_write> out_f: vec4<f32>;
@group(0) @binding(4) var<storage, read_write> out_b: vec4<i32>;

@compute @workgroup_size(1)
fn main() {
    let icond = inputs.ia < inputs.ib;

    // Operands are computed expressions on purpose: without baking they would
    // be re-emitted once per component.
    out_i = select(inputs.ia * 2, inputs.ib + 7, icond);

    out_u = select(inputs.ua, inputs.ub, inputs.ua < inputs.ub);

    // Float stays on `mix` at every version.
    out_f = select(inputs.fa, inputs.fb, inputs.fa < inputs.fb);

    // vec4<bool> operands as well as a vec4<bool> condition.
    let picked = select(icond, inputs.ua < inputs.ub, icond);
    out_b = select(vec4<i32>(0), vec4<i32>(1), picked);
}
