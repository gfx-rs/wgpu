// Immediates that contain `f16` values. SPIR-V stores them as `u32` words, so
// that Vulkan does not need `storagePushConstant16`.
enable f16;

struct Inner {
    h: f16,
    i: i32,
}

struct Data {
    a: f16,
    b: f16,
    v3: vec3<f16>,
    m: mat2x3<f16>,
    inner: Inner,
    f: f32,
    u: u32,
}

var<immediate> data: Data;

@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) i: u32) {
    let whole = data;
    out[0] = f32(whole.a);
    out[1] = f32(data.b);
    out[2] = f32(data.v3.z);
    out[3] = f32(data.v3[i]);
    let column = data.m[1];
    out[4] = f32(column.y);
    out[5] = f32(data.m[i][2]);
    out[6] = f32(data.inner.h) + f32(data.inner.i);
    out[7] = data.f + f32(data.u);
}
