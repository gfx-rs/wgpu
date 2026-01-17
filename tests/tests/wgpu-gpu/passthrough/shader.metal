#include <metal_stdlib>
using namespace metal;

/* Vertex output / Fragment input */
struct VSOut {
    float4 position [[position]];
};

/* Vertex shader */
vertex VSOut vertex_main(uint vid [[vertex_id]]) {
    VSOut out;

    // Hardcoded triangle in clip space
    float2 positions[3] = {
        float2( 0.0,  0.5),
        float2(-0.5, -0.5),
        float2( 0.5, -0.5),
    };

    out.position = float4(positions[vid], 0.0, 1.0);
    return out;
}

/* Fragment shader */
fragment float4 fragment_main(VSOut in [[stage_in]]) {
    // Solid white
    return float4(1.0, 1.0, 1.0, 1.0);
}
