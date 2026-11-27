// Regression test for <https://github.com/gfx-rs/wgpu/issues/10046>.

struct Inner { a: atomic<u32> }
struct Outer { inner: Inner }
@group(0) @binding(0) var<storage, read_write> s: Outer;
fn f() {
    let x = s;
}
