#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

struct ProbeArguments {
    command_buffer icb [[id(0)]];
};

struct VertexOutput {
    float4 position [[position]];
};

kernel void probe_generate(device ProbeArguments& arguments [[buffer(0)]]) {
    render_command command(arguments.icb, 0);
    command.draw_primitives(primitive_type::triangle, 0, 3, 1, 0);
}

vertex VertexOutput probe_vertex(uint vertex_id [[vertex_id]]) {
    float2 position;
    if (vertex_id == 0) {
        position = float2(-1.0, -1.0);
    } else if (vertex_id == 1) {
        position = float2(3.0, -1.0);
    } else {
        position = float2(-1.0, 3.0);
    }
    return { float4(position, 0.0, 1.0) };
}

fragment float4 probe_fragment() {
    return float4(1.0, 0.0, 0.0, 1.0);
}
