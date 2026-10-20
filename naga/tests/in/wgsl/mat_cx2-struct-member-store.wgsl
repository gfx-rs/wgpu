// Stores to `matCx2` members of structs that are not in a storage buffer.
//
// The HLSL backend splits such members into separate `vec2` fields. Stores to
// them must reach the caller's variable, and must only write the columns being
// stored (so a store to a `var<workgroup>` struct doesn't write back the rest).
//
// See https://github.com/gfx-rs/wgpu/issues/10658.

struct R {
    M: mat2x2<f32>,
    Minv: mat2x2<f32>,
}

struct Wide {
    a: mat4x2<f32>,
    b: mat3x2<f32>,
}

struct Outer {
    inner: R,
    arr: array<Wide, 2>,
}

struct WithMatArray {
    am: array<mat4x2<f32>, 2>,
}

@group(0) @binding(0)
var<storage, read_write> o: array<f32>;

var<private> p: R;
var<workgroup> w: Wide;

// The case from the issue: a function-local struct whose members are assigned.
fn make(a: f32) -> R {
    var r: R;
    r.M = mat2x2<f32>(1.0, 2.0, 3.0, a);
    r.Minv = mat2x2<f32>(5.0, 6.0, 7.0, 8.0);
    return r;
}

fn store_through_pointer(r: ptr<function, R>, x: f32) {
    (*r).M = mat2x2<f32>(x, x, x, x);
}

fn local_stores(idx: i32) -> Wide {
    var t: Wide;
    // whole matrix
    t.a = mat4x2<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    t.b = mat3x2<f32>(9.0, 10.0, 11.0, 12.0, 13.0, 14.0);
    // column with a dynamic index
    t.a[idx] = vec2<f32>(20.0, 21.0);
    // element with a dynamic column index
    t.b[idx][1] = 30.0;
    // element with dynamic column and row indices
    t.b[idx][idx] = 31.0;
    return t;
}

fn nested_stores(idx: i32) -> Outer {
    var n: Outer;
    n.inner.M = mat2x2<f32>(1.0, 2.0, 3.0, 4.0);
    n.inner.Minv[idx] = vec2<f32>(5.0, 6.0);
    n.arr[idx].a = mat4x2<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    n.arr[1].b[idx] = vec2<f32>(9.0, 10.0);
    n.arr[0].b[idx][idx] = 11.0;
    store_through_pointer(&n.inner, 12.0);
    return n;
}

fn mat_array_stores(idx: i32) -> WithMatArray {
    var m: WithMatArray;
    m.am[0][idx] = vec2<f32>(1.0, 2.0);
    m.am[1][idx][idx] = 3.0;
    return m;
}

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) li: u32) {
    let idx = i32(li);

    let r = make(0.5);
    o[0] = r.M[0][0];
    o[1] = r.Minv[1][1];

    let t = local_stores(idx);
    o[2] = t.a[0][0];
    o[3] = t.b[0][1];

    let n = nested_stores(idx);
    o[4] = n.inner.M[0][0];
    o[5] = n.arr[0].b[0][0];

    let m = mat_array_stores(idx);
    o[6] = m.am[1][0][0];

    p.M = mat2x2<f32>(1.0, 2.0, 3.0, 4.0);
    p.Minv[idx] = vec2<f32>(5.0, 6.0);
    o[7] = p.M[0][0] + p.Minv[0][0];

    w.a = mat4x2<f32>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    w.b[idx][idx] = 9.0;
    workgroupBarrier();
    o[8] = w.a[0][0] + w.b[0][0];
}
