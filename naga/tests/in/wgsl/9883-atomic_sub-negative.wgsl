var<workgroup> a: atomic<i32>;

@compute @workgroup_size(1)
fn main() {
    //  In GLSL, we accidentally printed `atomicAdd(a, --1)`. Oops!
    let x = atomicSub(&a, -1i);
}
