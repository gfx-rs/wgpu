@group(0) @binding(0)
var<storage,read_write> atomic_i32: atomic<i32>;
@group(0) @binding(1)
var<storage,read_write> atomic_u32: atomic<u32>;

fn test_atomic_i32() {
  atomicStore(&atomic_i32, 1);
  _ = atomicCompareExchangeWeak(&atomic_i32, 1, 1i);
  _ = atomicCompareExchangeWeak(&atomic_i32, 1i, 1);

  _ = atomicAdd(&atomic_i32, 1);
  _ = atomicSub(&atomic_i32, 1);
  _ = atomicAnd(&atomic_i32, 1);
  _ = atomicXor(&atomic_i32, 1);
  _ = atomicOr(&atomic_i32, 1);
  _ = atomicMin(&atomic_i32, 1);
  _ = atomicMax(&atomic_i32, 1);
  _ = atomicExchange(&atomic_i32, 1);
}

fn test_atomic_u32() {
  atomicStore(&atomic_u32, 1);
  _ = atomicCompareExchangeWeak(&atomic_u32, 1, 1u);
  _ = atomicCompareExchangeWeak(&atomic_u32, 1u, 1);

  _ = atomicAdd(&atomic_u32, 1);
  _ = atomicSub(&atomic_u32, 1);
  _ = atomicAnd(&atomic_u32, 1);
  _ = atomicXor(&atomic_u32, 1);
  _ = atomicOr(&atomic_u32, 1);
  _ = atomicMin(&atomic_u32, 1);
  _ = atomicMax(&atomic_u32, 1);
  _ = atomicExchange(&atomic_u32, 1);
}

@compute @workgroup_size(1)
fn main() {
    test_atomic_i32();
    test_atomic_u32();
}
