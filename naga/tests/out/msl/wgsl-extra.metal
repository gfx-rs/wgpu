// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct ImmediateData {
    uint index;
    char _pad1[4];
    metal::float2 double_;
};
struct FragmentIn {
    metal::float4 color;
    uint primitive_index;
    char _pad2[12];
};

struct main_Input {
    metal::float4 color [[user(loc0), center_perspective]];
};
struct main_Output {
    metal::float4 member [[color(0)]];
};
fragment main_Output main_(
  main_Input varyings [[stage_in]]
, uint primitive_index [[primitive_id]]
, constant ImmediateData& im [[buffer(1)]]
) {
    const FragmentIn in = { varyings.color, primitive_index };
    uint _e4 = im.index;
    if (in.primitive_index == _e4) {
        return main_Output { in.color };
    } else {
        metal::float3 _e11 = metal::float3(1.0) - in.color.xyz;
        return main_Output { metal::float4(_e11.x, _e11.y, _e11.z, in.color.w) };
    }
}
