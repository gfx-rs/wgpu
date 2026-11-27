// Concretizing an abstract array (here, indexing it with a non-constant
// index forces concretization to a concrete element type) invokes the
// layouter to compute the new array's stride. If a previously declared
// type is oversize, the layouter encounters it and the error must be
// reported rather than panicking.
//
// Regression test for <https://github.com/gfx-rs/wgpu/issues/9440>.

const a = array(0.);
var<workgroup> big: array<u32, 1 << 29>;
fn main(i: u32) {
    let x = a[i];
}
