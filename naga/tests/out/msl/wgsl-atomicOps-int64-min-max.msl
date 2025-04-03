// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    metal::atomic_ulong inner[2];
};
struct Struct {
    metal::atomic_ulong atomic_scalar;
    type_1 atomic_arr;
};

struct cs_mainInput {
};
kernel void cs_main(
  metal::uint3 id [[thread_position_in_threadgroup]]
, device metal::atomic_ulong& storage_atomic_scalar [[user(fake0)]]
, device type_1& storage_atomic_arr [[user(fake0)]]
, device Struct& storage_struct [[user(fake0)]]
, constant ulong& input [[user(fake0)]]
) {
    ulong _e3 = input;
    metal::atomic_max_explicit(&storage_atomic_scalar, _e3, metal::memory_order_relaxed);
    ulong _e7 = input;
    metal::atomic_max_explicit(&storage_atomic_arr.inner[1], 1uL + _e7, metal::memory_order_relaxed);
    metal::atomic_max_explicit(&storage_struct.atomic_scalar, 1uL, metal::memory_order_relaxed);
    metal::atomic_max_explicit(&storage_struct.atomic_arr.inner[1], static_cast<ulong>(id.x), metal::memory_order_relaxed);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    ulong _e20 = input;
    metal::atomic_min_explicit(&storage_atomic_scalar, _e20, metal::memory_order_relaxed);
    ulong _e24 = input;
    metal::atomic_min_explicit(&storage_atomic_arr.inner[1], 1uL + _e24, metal::memory_order_relaxed);
    metal::atomic_min_explicit(&storage_struct.atomic_scalar, 1uL, metal::memory_order_relaxed);
    metal::atomic_min_explicit(&storage_struct.atomic_arr.inner[1], static_cast<ulong>(id.x), metal::memory_order_relaxed);
    return;
}
