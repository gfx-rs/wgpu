// Used for metal and metallib passthrough

#include <metal_stdlib>
using namespace metal;

struct VSOut {
    float4 position [[position]];
};

vertex VSOut vertex_main(uint vid [[vertex_id]]) {
    VSOut out;

    float2 positions[3] = {
        float2( 0.0,  0.5),
        float2(-0.5, -0.5),
        float2( 0.5, -0.5),
    };

    out.position = float4(positions[vid], 0.0, 1.0);
    return out;
}

fragment float4 fragment_main(VSOut in [[stage_in]]) {
    return float4(1.0, 1.0, 1.0, 1.0);
}
