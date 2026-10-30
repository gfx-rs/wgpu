// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct main_Input {
};
struct main_Output {
    float member [[depth(greater)]];
};
fragment main_Output main_(
  metal::float4 pos [[position]]
) {
    return main_Output { pos.z + 0.1 };
}
