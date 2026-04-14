// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;


struct compInput {
};
kernel void comp(
  metal::uint3 id [[thread_position_in_grid]]
) {
    if (id.x == 0u) {
    }
    return;
}
