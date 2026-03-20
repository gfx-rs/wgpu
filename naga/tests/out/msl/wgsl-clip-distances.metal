// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_2 {
    float inner[1];
};
struct VertexOutput {
    metal::float4 position;
    type_2 clip_distances;
    char _pad2[12];
};

struct main_Output {
    metal::float4 position [[position]];
    float clip_distances [[clip_distance]] [1];
};
vertex main_Output main_(
) {
    VertexOutput out = {};
    out.clip_distances.inner[0] = 0.5;
    VertexOutput _e4 = out;
    const auto _tmp = _e4;
    return main_Output { _tmp.position, {_tmp.clip_distances.inner[0]} };
}
