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

fn make(a: f32) -> R {
    var r: R;

    r.M = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, a));
    r.Minv = mat2x2<f32>(vec2<f32>(5f, 6f), vec2<f32>(7f, 8f));
    let _e17 = r;
    return _e17;
}

fn store_through_pointer(r_1: ptr<function, R>, x: f32) {
    (*r_1).M = mat2x2<f32>(vec2<f32>(x, x), vec2<f32>(x, x));
    return;
}

fn local_stores(idx: i32) -> Wide {
    var t: Wide;

    t.a = mat4x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f), vec2<f32>(5f, 6f), vec2<f32>(7f, 8f));
    t.b = mat3x2<f32>(vec2<f32>(9f, 10f), vec2<f32>(11f, 12f), vec2<f32>(13f, 14f));
    t.a[idx] = vec2<f32>(20f, 21f);
    t.b[idx][1] = 30f;
    t.b[idx][idx] = 31f;
    let _e40 = t;
    return _e40;
}

fn nested_stores(idx_1: i32) -> Outer {
    var n: Outer;

    n.inner.M = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f));
    n.inner.Minv[idx_1] = vec2<f32>(5f, 6f);
    n.arr[idx_1].a = mat4x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f), vec2<f32>(5f, 6f), vec2<f32>(7f, 8f));
    n.arr[1].b[idx_1] = vec2<f32>(9f, 10f);
    n.arr[0].b[idx_1][idx_1] = 11f;
    store_through_pointer((&n.inner), 12f);
    let _e48 = n;
    return _e48;
}

fn mat_array_stores(idx_2: i32) -> WithMatArray {
    var m: WithMatArray;

    m.am[0][idx_2] = vec2<f32>(1f, 2f);
    m.am[1][idx_2][idx_2] = 3f;
    let _e13 = m;
    return _e13;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(local_invocation_index) li: u32) {
    let idx_3 = i32(li);
    let _e3 = make(0.5f);
    o[0] = _e3.M[0].x;
    o[1] = _e3.Minv[1].y;
    let _e14 = local_stores(idx_3);
    o[2] = _e14.a[0].x;
    o[3] = _e14.b[0].y;
    let _e25 = nested_stores(idx_3);
    o[4] = _e25.inner.M[0].x;
    o[5] = _e25.arr[0].b[0].x;
    let _e39 = mat_array_stores(idx_3);
    o[6] = _e39.am[1][0].x;
    p.M = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f));
    p.Minv[idx_3] = vec2<f32>(5f, 6f);
    let _e67 = p.M[0][0];
    let _e72 = p.Minv[0][0];
    o[7] = (_e67 + _e72);
    w.a = mat4x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f), vec2<f32>(5f, 6f), vec2<f32>(7f, 8f));
    w.b[idx_3][idx_3] = 9f;
    workgroupBarrier();
    let _e100 = w.a[0][0];
    let _e105 = w.b[0][0];
    o[8] = (_e100 + _e105);
    return;
}
