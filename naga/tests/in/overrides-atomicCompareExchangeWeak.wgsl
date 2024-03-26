override o: i32;
var<workgroup> a: atomic<u32>;

@compute @workgroup_size(1)
fn f() {
  atomicCompareExchangeWeak(&a, u32(o), 1u);
}
