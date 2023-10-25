// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Vertex {
    metal::float2 position;
};
struct NoteInstance {
    metal::float2 position;
};
struct VertexOutput {
    metal::float4 position;
};

struct vs_mainInput {
    metal::float2 position [[attribute(0)]];
    metal::float2 position_1 [[attribute(1)]];
};
struct vs_mainOutput {
    metal::float4 position [[position]];
};
vertex vs_mainOutput vs_main(
  vs_mainInput varyings [[stage_in]]
) {
    const Vertex vertex_ = { varyings.position };
    const NoteInstance note = { varyings.position_1 };
    VertexOutput out = {};
    VertexOutput _e3 = out;
    const auto _tmp = _e3;
    return vs_mainOutput { _tmp.position };
}


struct fs_mainInput {
    metal::float2 position [[user(loc1), center_perspective]];
};
struct fs_mainOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment fs_mainOutput fs_main(
  fs_mainInput varyings_1 [[stage_in]]
, metal::float4 position [[position]]
) {
    const VertexOutput in = { position };
    const NoteInstance note_1 = { varyings_1.position };
    metal::float3 position_1 = metal::float3(1.0);
    return fs_mainOutput { in.position };
}
