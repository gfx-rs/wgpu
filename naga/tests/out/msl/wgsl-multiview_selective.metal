// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct FragmentIn {
    metal::float4 position;
    uint view_index;
    char _pad2[12];
};

struct vs_mainInput {
};
struct vs_mainOutput {
    metal::float4 member [[position]];
};
vertex vs_mainOutput vs_main(
  uint amplification_id [[amplification_id]]
) {
    const uint view_index_map[2] = {0u, 3u};
    const uint view_index = view_index_map[amplification_id];
    return vs_mainOutput { metal::float4(static_cast<float>(view_index)) };
}


struct fs_mainInput {
};
struct fs_mainOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment fs_mainOutput fs_main(
  metal::float4 position [[position]]
, uint amplification_id_1 [[amplification_id]]
) {
    const uint view_index_map_1[2] = {0u, 3u};
    const uint view_index_1 = view_index_map_1[amplification_id_1];
    const FragmentIn in = { position, view_index_1 };
    return fs_mainOutput { metal::float4(static_cast<float>(in.view_index)) };
}
