// Regression test for <https://github.com/gfx-rs/wgpu/issues/8612>.
//
// The Vulkan environment spec (`VUID-RuntimeSpirv-None-04745`) requires
// that accesses into a `PushConstant`-storage-class variable that are
// arrays use dynamically uniform indices; this restriction also applies
// to vectors. So naga must not emit `OpAccessChain` with a non-constant
// index directly into a vector living in the `immediate` (push-constant)
// address space -- it must instead load the whole vector and use
// `OpVectorExtractDynamic`.
//
// Naming (mirrors `mat_cx2.wgsl`):
// V = vector field, M = matrix field, C = constant index, trailing
// C/V = constant/variable (dynamic) index into the vector/column.

struct Immediates {
    v: vec4<f32>,
    m: mat4x4<f32>,
}

var<immediate> im: Immediates;

@group(0) @binding(0)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) idx: u32) {
    // Dynamically indexing a vector field directly. `im.v` is a
    // `TypeInner::Pointer` to a vector (it has a concrete entry in the
    // module's type arena, being a struct field).
    let v_c = im.v[0];
    let v_v = im.v[idx];

    // Dynamically indexing a component of a matrix column. `im.m[0]` is a
    // `TypeInner::ValuePointer` to a vector (it doesn't have its own type
    // arena entry), which is a distinct code path from `v_v` above.
    let m_cc = im.m[0][0];
    let m_cv = im.m[0][idx];

    out[0] = v_c;
    out[1] = v_v;
    out[2] = m_cc;
    out[3] = m_cv;
}
