// Regression test for <https://github.com/gfx-rs/wgpu/issues/10046>.

struct S { a: atomic<u32> }
@group(0) @binding(0) var<storage, read_write> s: S;
fn f() {
    _ = s;
}
