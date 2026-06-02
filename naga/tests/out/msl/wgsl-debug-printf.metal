// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct main_Input {
};
kernel void main_(
  metal::uint3 id [[thread_position_in_grid]]
) {
    metal::os_log_default.log_info("debug id: %u %u %u", id.x, id.y, id.z);
    return;
}
