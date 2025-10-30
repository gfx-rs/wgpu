// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct fs_mainInput {
};
struct fs_mainOutput {
    metal::float4 member [[color(0)]];
};
fragment fs_mainOutput fs_main(
  metal::float3 bary [[barycentric_coord]]
) {
    return fs_mainOutput { metal::float4(bary, 1.0) };
}
