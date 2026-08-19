// language: metal2.1
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct StructDepthOutput {
    float depth;
};

struct greaterOutput {
    float member [[depth(greater)]];
};
fragment greaterOutput greater(
) {
    return greaterOutput { 0.5 };
}


struct lessOutput {
    float member_1 [[depth(less)]];
};
fragment lessOutput less(
) {
    return lessOutput { 0.5 };
}


struct plainOutput {
    float member_2 [[depth(any)]];
};
fragment plainOutput plain(
) {
    return plainOutput { 0.5 };
}


struct struct_greaterOutput {
    float depth [[depth(greater)]];
};
fragment struct_greaterOutput struct_greater(
) {
    const auto _tmp = StructDepthOutput {0.5};
    return struct_greaterOutput { _tmp.depth };
}
