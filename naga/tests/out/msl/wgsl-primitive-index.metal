// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct funcInput {
};
struct funcOutput {
    metal::float4 member [[color(0)]];
};
fragment funcOutput func(
  uint index [[primitive_id]]
) {
    return funcOutput { metal::float4(static_cast<float>(index), 1.0, 1.0, 1.0) };
}
