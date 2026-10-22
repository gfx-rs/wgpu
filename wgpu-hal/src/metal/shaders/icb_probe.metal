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

// Both stages read buffers the ICB command inherits, at the slots wgpu uses:
// the first vertex buffer sits at the top of the argument table (30) and the
// first bind-group buffer at 0. A device that faults on an inherited buffer
// the ICB descriptor's bind counts leave out fails the probe, not a draw.
vertex VertexOutput probe_vertex(
    uint vertex_id [[vertex_id]],
    const device float2* positions [[buffer(30)]])
{
    return { float4(positions[vertex_id], 0.0, 1.0) };
}

fragment float4 probe_fragment(constant float4& color [[buffer(0)]]) {
    return color;
}
