// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

typedef uint type_1[1];

kernel void main_(
  device type_1& data [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    uint i = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e12 = i;
            i = _e12 + 1u;
        }
        loop_init = false;
        uint _e2 = i;
        if (_e2 < 4u) {
        } else {
            break;
        }
        {
            uint _e6 = i;
            uint _e8 = i;
            data[_e6] = _e8 * 2u;
        }
    }
    return;
}
