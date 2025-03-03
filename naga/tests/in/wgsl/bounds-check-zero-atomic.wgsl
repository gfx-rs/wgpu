// Tests for `naga::back::BoundsCheckPolicy::ReadZeroSkipWrite` for atomic types.

// These are separate from `bounds-check-zero.wgsl because SPIR-V does not yet
// support `ReadZeroSkipWrite` for atomics. Once it does, the test files could
// be combined.

struct Globals {
    a: atomic<u32>,
    b: array<atomic<u32>, 10>,
    c: array<atomic<u32>>,
}

@group(0) @binding(0) var<storage, read_write> globals: Globals;

fn fetch_add_atomic() -> u32 {
   return atomicAdd(&globals.a, 1u);
}

fn fetch_add_atomic_static_sized_array(i: i32) -> u32 {
   return atomicAdd(&globals.b[i], 1u);
}

fn fetch_add_atomic_dynamic_sized_array(i: i32) -> u32 {
   return atomicAdd(&globals.c[i], 1u);
}

fn exchange_atomic() -> u32 {
   return atomicExchange(&globals.a, 1u);
}

fn exchange_atomic_static_sized_array(i: i32) -> u32 {
   return atomicExchange(&globals.b[i], 1u);
}

fn exchange_atomic_dynamic_sized_array(i: i32) -> u32 {
   return atomicExchange(&globals.c[i], 1u);
}

fn fetch_add_atomic_dynamic_sized_array_static_index() -> u32 {
   return atomicAdd(&globals.c[1000], 1u);
}

fn exchange_atomic_dynamic_sized_array_static_index() -> u32 {
   return atomicExchange(&globals.c[1000], 1u);
}

