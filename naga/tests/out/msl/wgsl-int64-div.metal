// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

ulong naga_div(ulong lhs, ulong rhs) {
    return lhs / metal::select(rhs, 1uL, rhs == 0uL);
}

ulong naga_mod(ulong lhs, ulong rhs) {
    return lhs % metal::select(rhs, 1uL, rhs == 0uL);
}

long naga_div(long lhs, long rhs) {
    return lhs / metal::select(rhs, 1L, (lhs == (-9223372036854775807L - 1L) & rhs == -1L) | (rhs == 0L));
}

long naga_mod(long lhs, long rhs) {
    long divisor = metal::select(rhs, 1L, (lhs == (-9223372036854775807L - 1L) & rhs == -1L) | (rhs == 0L));
    return lhs - (lhs / divisor) * divisor;
}

metal::ulong2 naga_div(metal::ulong2 lhs, metal::ulong2 rhs) {
    return lhs / metal::select(rhs, 1uL, rhs == 0uL);
}

metal::ulong2 naga_mod(metal::ulong2 lhs, metal::ulong2 rhs) {
    return lhs % metal::select(rhs, 1uL, rhs == 0uL);
}

metal::long2 naga_div(metal::long2 lhs, metal::long2 rhs) {
    return lhs / metal::select(rhs, 1L, (lhs == (-9223372036854775807L - 1L) & rhs == -1L) | (rhs == 0L));
}

metal::long2 naga_mod(metal::long2 lhs, metal::long2 rhs) {
    metal::long2 divisor = metal::select(rhs, 1L, (lhs == (-9223372036854775807L - 1L) & rhs == -1L) | (rhs == 0L));
    return lhs - (lhs / divisor) * divisor;
}


[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  device ulong& out_u64_ [[user(fake0)]]
, device long& out_i64_ [[user(fake0)]]
, device metal::ulong2& out_vec_u64_ [[user(fake0)]]
, device metal::long2& out_vec_i64_ [[user(fake0)]]
) {
    out_u64_ = naga_div(10uL, 3uL) + naga_mod(10uL, 3uL);
    out_i64_ = as_type<long>(as_type<ulong>(naga_div(-10L, 3L)) + as_type<ulong>(naga_mod(-10L, 3L)));
    metal::ulong2 va = metal::ulong2(10uL);
    metal::ulong2 vb = metal::ulong2(3uL);
    metal::long2 vc = metal::long2(-10L);
    metal::long2 vd = metal::long2(3L);
    out_vec_u64_ = naga_div(va, vb) + naga_mod(va, vb);
    out_vec_i64_ = as_type<metal::long2>(as_type<metal::ulong2>(naga_div(vc, vd)) + as_type<metal::ulong2>(naga_mod(vc, vd)));
    return;
}
