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
struct type_3 {
    float inner[6];
};
struct VertexOutputSplit {
    metal::float4 color;
    type_3 clip_distances;
    char _pad2[8];
    metal::float4 position;
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


struct main_splitOutput {
    metal::float4 color [[user(loc0), center_perspective]];
    float clip_distances [[clip_distance]] [6];
    metal::float4 position [[position]];
};
vertex main_splitOutput main_split(
) {
    VertexOutputSplit out_1 = {};
    int i = 0;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            int _e14 = i;
            i = as_type<int>(as_type<uint>(_e14) + as_type<uint>(1));
        }
        loop_init = false;
        int _e3 = i;
        if (_e3 < 6) {
        } else {
            break;
        }
        {
            int _e7 = i;
            int _e9 = i;
            out_1.clip_distances.inner[_e7] = static_cast<float>(_e9) - 2.5;
        }
    }
    VertexOutputSplit _e16 = out_1;
    const auto _tmp_1 = _e16;
    return main_splitOutput { _tmp_1.color, {_tmp_1.clip_distances.inner[0],_tmp_1.clip_distances.inner[1],_tmp_1.clip_distances.inner[2],_tmp_1.clip_distances.inner[3],_tmp_1.clip_distances.inner[4],_tmp_1.clip_distances.inner[5]}, _tmp_1.position };
}
