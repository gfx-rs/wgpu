// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct fs_mainInput {
    metal::float3 bary [[barycentric_coord]];
};
struct fs_mainOutput {
    metal::float4 member [[color(0)]];
};
fragment fs_mainOutput fs_main(
  fs_mainInput varyings [[stage_in]]
) {
    const auto bary = varyings.bary;
    return fs_mainOutput { metal::float4(bary, 1.0) };
}


struct fs_main_no_perspectiveInput {
    metal::float3 bary_1 [[barycentric_coord, center_no_perspective]];
};
struct fs_main_no_perspectiveOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment fs_main_no_perspectiveOutput fs_main_no_perspective(
  fs_main_no_perspectiveInput varyings_1 [[stage_in]]
) {
    const auto bary_1 = varyings_1.bary_1;
    return fs_main_no_perspectiveOutput { metal::float4(bary_1, 1.0) };
}
