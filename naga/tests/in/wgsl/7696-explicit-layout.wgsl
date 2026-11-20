// Only types used by buffers get explicit layout decorations.

struct Local {
    a: vec2<f32>,
    b: f32,
}

struct InBuffer {
    m: mat2x2<f32>,
    v: array<vec4<f32>, 2>,
}

@group(0) @binding(0) var<storage, read_write> buf: InBuffer;
@group(0) @binding(1) var<uniform> uni: InBuffer;

var<private> p: array<u32, 3>;
var<workgroup> w: array<i32, 5>;

@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) i: u32) {
    var f = array<vec2<f32>, 6>(vec2(-1.0), vec2(1.0), vec2(0.0), vec2(2.0), vec2(3.0), vec2(4.0));
    var l = Local(f[i % 6u], 1.0);
    p[i % 3u] = i;
    w[i % 5u] = i32(i);
    workgroupBarrier();
    buf.v[0] = vec4(l.a, l.b, f32(p[0]) + f32(w[1])) + uni.v[0];
}
