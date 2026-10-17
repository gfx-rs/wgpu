// Compute kernels that translate `DrawIndirectArgs`/`DrawIndexedIndirectArgs`
// sequences into Metal indirect-command-buffer (ICB) render commands on the
// GPU, so `multi_draw_(indexed_)indirect(_count)` never round-trips draw
// arguments through the CPU.
//
// The Rust side (`wgpu-hal/src/metal/command.rs`) binds the ICB through an
// argument buffer (`WgpuIcbArguments`) and dispatches one thread per potential
// draw. Draws with a zero-sized dimension are `reset()` so the slot becomes a
// no-op, matching WebGPU's treatment of empty draws.

#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

struct WgpuIcbArguments {
    command_buffer icb [[id(0)]];
};

// Must match the layout of `wgpu_types::DrawIndirectArgs`.
struct WgpuDrawIndirectArgs {
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

// Must match the layout of `wgpu_types::DrawIndexedIndirectArgs`.
struct WgpuDrawIndexedIndirectArgs {
    uint index_count;
    uint instance_count;
    uint first_index;
    int base_vertex;
    uint first_instance;
};

// Must match the layout of `MTLIndirectCommandBufferExecutionRange`.
struct WgpuIcbExecutionRange {
    uint location;
    uint length;
};

// Must match the `ICB_PRIMITIVE_*` constants in
// `wgpu-hal/src/metal/command.rs`.
enum WgpuIcbPrimitiveType : uint {
    WGPU_ICB_PRIMITIVE_POINT = 0,
    WGPU_ICB_PRIMITIVE_LINE = 1,
    WGPU_ICB_PRIMITIVE_LINE_STRIP = 2,
    WGPU_ICB_PRIMITIVE_TRIANGLE = 3,
    WGPU_ICB_PRIMITIVE_TRIANGLE_STRIP = 4,
};

static primitive_type wgpu_icb_primitive_type(uint value) {
    switch (value) {
        case WGPU_ICB_PRIMITIVE_POINT:
            return primitive_type::point;
        case WGPU_ICB_PRIMITIVE_LINE:
            return primitive_type::line;
        case WGPU_ICB_PRIMITIVE_LINE_STRIP:
            return primitive_type::line_strip;
        case WGPU_ICB_PRIMITIVE_TRIANGLE:
            return primitive_type::triangle;
        case WGPU_ICB_PRIMITIVE_TRIANGLE_STRIP:
            return primitive_type::triangle_strip;
        default:
            return primitive_type::triangle;
    }
}

kernel void wgpu_generate_mdi_icb(
    device WgpuIcbArguments& icb_args [[buffer(0)]],
    const device WgpuDrawIndirectArgs* draw_args [[buffer(1)]],
    constant uint& primitive_type_value [[buffer(2)]],
    constant uint& draw_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= draw_count) {
        return;
    }
    const WgpuDrawIndirectArgs args = draw_args[tid];
    render_command cmd(icb_args.icb, tid);
    if (args.vertex_count == 0 || args.instance_count == 0) {
        cmd.reset();
        return;
    }
    const primitive_type primitive = wgpu_icb_primitive_type(primitive_type_value);
    cmd.draw_primitives(
        primitive,
        args.first_vertex,
        args.vertex_count,
        args.instance_count,
        args.first_instance);
}

kernel void wgpu_generate_indexed_mdi_icb_u16(
    device WgpuIcbArguments& icb_args [[buffer(0)]],
    const device WgpuDrawIndexedIndirectArgs* draw_args [[buffer(1)]],
    const device ushort* index_buffer [[buffer(2)]],
    constant uint& primitive_type_value [[buffer(3)]],
    constant uint& draw_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= draw_count) {
        return;
    }
    const WgpuDrawIndexedIndirectArgs args = draw_args[tid];
    render_command cmd(icb_args.icb, tid);
    if (args.index_count == 0 || args.instance_count == 0) {
        cmd.reset();
        return;
    }
    const primitive_type primitive = wgpu_icb_primitive_type(primitive_type_value);
    cmd.draw_indexed_primitives(
        primitive,
        args.index_count,
        index_buffer + args.first_index,
        args.instance_count,
        args.base_vertex,
        args.first_instance);
}

kernel void wgpu_generate_indexed_mdi_icb_u32(
    device WgpuIcbArguments& icb_args [[buffer(0)]],
    const device WgpuDrawIndexedIndirectArgs* draw_args [[buffer(1)]],
    const device uint* index_buffer [[buffer(2)]],
    constant uint& primitive_type_value [[buffer(3)]],
    constant uint& draw_count [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= draw_count) {
        return;
    }
    const WgpuDrawIndexedIndirectArgs args = draw_args[tid];
    render_command cmd(icb_args.icb, tid);
    if (args.index_count == 0 || args.instance_count == 0) {
        cmd.reset();
        return;
    }
    const primitive_type primitive = wgpu_icb_primitive_type(primitive_type_value);
    cmd.draw_indexed_primitives(
        primitive,
        args.index_count,
        index_buffer + args.first_index,
        args.instance_count,
        args.base_vertex,
        args.first_instance);
}

// Clamps a GPU-resident draw count to `max_draw_count` and writes the result
// as the ICB execution range used by
// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`, keeping
// `multi_draw_*_indirect_count` counts on the GPU.
kernel void wgpu_generate_mdi_execution_range(
    const device uint& draw_count [[buffer(0)]],
    device WgpuIcbExecutionRange& range [[buffer(1)]],
    constant uint& max_draw_count [[buffer(2)]])
{
    range.location = 0;
    range.length = min(draw_count, max_draw_count);
}

// Fallback for `multi_draw_*_indirect_count` when the bound pipeline can't
// execute inside an ICB (see `RenderPipeline::icb_raw`):
// copies `min(draw_count, max_draw_count)` argument structs into a private
// destination buffer and zeroes the rest, so a fixed `max_draw_count`-length
// loop of per-draw indirect calls executes exactly `draw_count` real draws.
// Operates on 4-byte words with `words_per_draw` set per argument type; a
// zeroed argument struct is a no-op draw for all three draw families.
kernel void wgpu_clamp_mdi_args(
    device uint* dst_args [[buffer(0)]],
    const device uint* src_args [[buffer(1)]],
    const device uint& draw_count [[buffer(2)]],
    constant uint& max_draw_count [[buffer(3)]],
    constant uint& words_per_draw [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    const uint draw_index = tid / words_per_draw;
    if (draw_index >= max_draw_count) {
        return;
    }
    const uint live_draws = min(draw_count, max_draw_count);
    dst_args[tid] = draw_index < live_draws ? src_args[tid] : 0u;
}
