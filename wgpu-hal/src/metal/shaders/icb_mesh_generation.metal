// Compute kernel that translates `DispatchIndirectArgs` sequences into Metal
// indirect-command-buffer (ICB) mesh-draw commands on the GPU, the
// mesh-shader counterpart of `icb_generation.metal`. Compiled lazily because
// mesh commands in ICBs need a newer OS baseline than plain draws.

#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

struct WgpuIcbArguments {
    command_buffer icb [[id(0)]];
};

// Must match the layout of `wgpu_types::DispatchIndirectArgs`.
struct WgpuDispatchIndirectArgs {
    uint x;
    uint y;
    uint z;
};

// Threadgroup sizes captured from the bound pipeline at encode time; a
// mesh-draw ICB command needs them alongside the indirect threadgroup counts.
struct WgpuMeshThreadgroupSizes {
    uint object_x;
    uint object_y;
    uint object_z;
    uint mesh_x;
    uint mesh_y;
    uint mesh_z;
};

kernel void wgpu_generate_mesh_mdi_icb(
    device WgpuIcbArguments& icb_args [[buffer(0)]],
    const device WgpuDispatchIndirectArgs* draw_args [[buffer(1)]],
    constant WgpuMeshThreadgroupSizes& sizes [[buffer(2)]],
    constant uint& draw_count [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= draw_count) {
        return;
    }
    const WgpuDispatchIndirectArgs args = draw_args[tid];
    render_command cmd(icb_args.icb, tid);
    if (args.x == 0 || args.y == 0 || args.z == 0) {
        cmd.reset();
        return;
    }
    cmd.draw_mesh_threadgroups(
        uint3(args.x, args.y, args.z),
        uint3(sizes.object_x, sizes.object_y, sizes.object_z),
        uint3(sizes.mesh_x, sizes.mesh_y, sizes.mesh_z));
}
