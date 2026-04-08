// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct VertexInput {
    metal::packed_int3 chunk;
    uint texture_index;
};
struct VertexOutput {
    metal::float4 clip_position;
};

struct vs_mainInput {
    metal::int3 chunk [[attribute(0)]];
    uint texture_index [[attribute(1)]];
};
struct vs_mainOutput {
    metal::float4 clip_position [[position]];
};
vertex vs_mainOutput vs_main(
  vs_mainInput varyings [[stage_in]]
) {
    const VertexInput in = { varyings.chunk, varyings.texture_index };
    VertexOutput out = {};
    metal::float3 position = static_cast<metal::float3>(as_type<metal::int3>(as_type<metal::uint3>(metal::int3(in.chunk)) - as_type<metal::uint3>(metal::int3(5))));
    VertexOutput _e7 = out;
    const auto _tmp = _e7;
    return vs_mainOutput { _tmp.clip_position };
}
