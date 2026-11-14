// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


struct main_Output {
    metal::float4 member [[color(0)]];
};
[[early_fragment_tests]] fragment main_Output main_(
) {
    return main_Output { metal::float4(0.4, 0.3, 0.2, 0.1) };
}
