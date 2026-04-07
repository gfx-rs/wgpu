// language: metal4.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    float inner[3];
};

struct fs_mainInput {
    metal::vertex_value<float> v [[user(loc0)]];
};
struct fs_mainOutput {
    metal::float4 member [[color(0)]];
};
fragment fs_mainOutput fs_main(
  fs_mainInput varyings [[stage_in]]
) {
    const type_1 v = { varyings.v.get(metal::vertex_index::first), varyings.v.get(metal::vertex_index::second), varyings.v.get(metal::vertex_index::third) };
    return fs_mainOutput { metal::float4(v.inner[0], v.inner[1], v.inner[2], 1.0) };
}
