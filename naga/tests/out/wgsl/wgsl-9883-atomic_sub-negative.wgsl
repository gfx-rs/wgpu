var<workgroup> a: atomic<i32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let _e2 = atomicSub((&a), -1i);
    return;
}
