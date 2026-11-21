// Regression test for <https://github.com/gfx-rs/wgpu/issues/7419>: a
// constructor component of the wrong concrete type used to be reported by the
// IR validator as `Composing 0's component type is not expected`.

struct S { inner: array<u32, 4> }
fn main() {
    var s = S(1u);
}
