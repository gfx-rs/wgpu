// An `array(...)` constructor expression invokes the layouter to compute
// the stride of the constructed array. If a previously declared type is
// oversize, the layouter encounters it and the error must be reported
// rather than panicking.
//
// Regression test for <https://github.com/gfx-rs/wgpu/issues/9440>.

var<workgroup> big: array<u32, 1 << 29>;
const A = array(1);
