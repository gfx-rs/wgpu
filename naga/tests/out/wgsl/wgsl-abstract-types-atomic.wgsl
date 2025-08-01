@group(0) @binding(0) 
var<storage, read_write> atomic_i32_: atomic<i32>;
@group(0) @binding(1) 
var<storage, read_write> atomic_u32_: atomic<u32>;

fn test_atomic_i32_() {
    atomicStore((&atomic_i32_), 1i);
    let _e5 = atomicCompareExchangeWeak((&atomic_i32_), 1i, 1i);
    let _e9 = atomicCompareExchangeWeak((&atomic_i32_), 1i, 1i);
    let _e12 = atomicAdd((&atomic_i32_), 1i);
    let _e15 = atomicSub((&atomic_i32_), 1i);
    let _e18 = atomicAnd((&atomic_i32_), 1i);
    let _e21 = atomicXor((&atomic_i32_), 1i);
    let _e24 = atomicOr((&atomic_i32_), 1i);
    let _e27 = atomicMin((&atomic_i32_), 1i);
    let _e30 = atomicMax((&atomic_i32_), 1i);
    let _e33 = atomicExchange((&atomic_i32_), 1i);
    return;
}

fn test_atomic_u32_() {
    atomicStore((&atomic_u32_), 1u);
    let _e5 = atomicCompareExchangeWeak((&atomic_u32_), 1u, 1u);
    let _e9 = atomicCompareExchangeWeak((&atomic_u32_), 1u, 1u);
    let _e12 = atomicAdd((&atomic_u32_), 1u);
    let _e15 = atomicSub((&atomic_u32_), 1u);
    let _e18 = atomicAnd((&atomic_u32_), 1u);
    let _e21 = atomicXor((&atomic_u32_), 1u);
    let _e24 = atomicOr((&atomic_u32_), 1u);
    let _e27 = atomicMin((&atomic_u32_), 1u);
    let _e30 = atomicMax((&atomic_u32_), 1u);
    let _e33 = atomicExchange((&atomic_u32_), 1u);
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    test_atomic_i32_();
    test_atomic_u32_();
    return;
}
