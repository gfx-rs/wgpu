// language: metal3.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    metal::atomic_float inner[2];
};
struct Struct {
    metal::atomic_float atomic_scalar;
    type_1 atomic_arr;
};

struct cs_mainInput {
};
kernel void cs_main(
  metal::uint3 id [[thread_position_in_threadgroup]]
, device metal::atomic_float& storage_atomic_scalar [[user(fake0)]]
, device type_1& storage_atomic_arr [[user(fake0)]]
, device Struct& storage_struct [[user(fake0)]]
) {
    metal::atomic_store_explicit(&storage_atomic_scalar, 1.5, metal::memory_order_relaxed);
    metal::atomic_store_explicit(&storage_atomic_arr.inner[1], 1.5, metal::memory_order_relaxed);
    metal::atomic_store_explicit(&storage_struct.atomic_scalar, 1.5, metal::memory_order_relaxed);
    metal::atomic_store_explicit(&storage_struct.atomic_arr.inner[1], 1.5, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float l0_ = metal::atomic_load_explicit(&storage_atomic_scalar, metal::memory_order_relaxed);
    float l1_ = metal::atomic_load_explicit(&storage_atomic_arr.inner[1], metal::memory_order_relaxed);
    float l2_ = metal::atomic_load_explicit(&storage_struct.atomic_scalar, metal::memory_order_relaxed);
    float l3_ = metal::atomic_load_explicit(&storage_struct.atomic_arr.inner[1], metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float _e27 = metal::atomic_fetch_add_explicit(&storage_atomic_scalar, 1.5, metal::memory_order_relaxed);
    float _e31 = metal::atomic_fetch_add_explicit(&storage_atomic_arr.inner[1], 1.5, metal::memory_order_relaxed);
    float _e35 = metal::atomic_fetch_add_explicit(&storage_struct.atomic_scalar, 1.5, metal::memory_order_relaxed);
    float _e40 = metal::atomic_fetch_add_explicit(&storage_struct.atomic_arr.inner[1], 1.5, metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    float _e43 = metal::atomic_exchange_explicit(&storage_atomic_scalar, 1.5, metal::memory_order_relaxed);
    float _e47 = metal::atomic_exchange_explicit(&storage_atomic_arr.inner[1], 1.5, metal::memory_order_relaxed);
    float _e51 = metal::atomic_exchange_explicit(&storage_struct.atomic_scalar, 1.5, metal::memory_order_relaxed);
    float _e56 = metal::atomic_exchange_explicit(&storage_struct.atomic_arr.inner[1], 1.5, metal::memory_order_relaxed);
    return;
}
