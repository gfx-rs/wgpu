// language: metal2.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    float inner[4];
};
struct type_3 {
    int inner[4];
};

kernel void main_(
  device type_1& floats [[user(fake0)]]
, device type_3& ints [[user(fake0)]]
) {
    float _e4 = floats.inner[1];
    floats.inner[0] = metal::sign(_e4);
    float _e8 = floats.inner[2];
    float _e11 = floats.inner[3];
    metal::float2 v = metal::float2(_e8, _e11);
    metal::float2 signs = metal::sign(v);
    floats.inner[2] = signs.x;
    floats.inner[3] = signs.y;
    int _e24 = ints.inner[1];
    ints.inner[0] = metal::select(metal::select(int(-1), int(1), (_e24 > 0)), int(0), (_e24 == 0));
    return;
}
