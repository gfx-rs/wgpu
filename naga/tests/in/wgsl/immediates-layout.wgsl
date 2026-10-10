// Immediates use the same layout as storage buffers: struct-typed members are not
// aligned to 16 bytes, so they can land at offsets that HLSL constant buffers would
// never pick on their own.

struct Inner {
    a: u32,
    b: u32,
}

// `inner` is at offset 4.
struct NestedAfterScalar {
    x: u32,
    inner: Inner,
    y: u32,
}

// `inner` is at offset 12.
struct NestedAfterVec3 {
    v: vec3<u32>,
    inner: Inner,
    y: u32,
}

struct Mixed {
    s: i32,
    m: mat3x2<f32>,
    inner: Inner,
    m4: mat4x3<f32>,
    v4: vec4<i32>,
    f: f32,
}

var<immediate> nested_after_scalar: NestedAfterScalar;
var<immediate> nested_after_vec3: NestedAfterVec3;
var<immediate> mixed: Mixed;
var<immediate> non_struct: vec4<f32>;

@group(0) @binding(0)
var<storage, read_write> out: array<u32>;

fn read_nested_after_vec3() -> u32 {
    return nested_after_vec3.inner.b;
}

@compute @workgroup_size(1)
fn scalar_then_struct() {
    out[0] = nested_after_scalar.x;
    out[1] = nested_after_scalar.inner.a;
    out[2] = nested_after_scalar.inner.b;
    out[3] = nested_after_scalar.y;
}

@compute @workgroup_size(1)
fn vec3_then_struct() {
    out[0] = nested_after_vec3.v.z;
    out[1] = nested_after_vec3.inner.a;
    out[2] = read_nested_after_vec3();
    out[3] = nested_after_vec3.y;
}

@compute @workgroup_size(1)
fn matrices_and_vectors() {
    out[0] = bitcast<u32>(mixed.s);
    out[1] = bitcast<u32>(mixed.m[1].y);
    out[2] = mixed.inner.b;
    out[3] = bitcast<u32>(mixed.m4[2].z);
    out[4] = bitcast<u32>(mixed.v4.w);
    out[5] = bitcast<u32>(mixed.f);
}

@compute @workgroup_size(1)
fn whole_vector() {
    out[0] = bitcast<u32>(non_struct.x);
    out[1] = bitcast<u32>(non_struct.w);
}
