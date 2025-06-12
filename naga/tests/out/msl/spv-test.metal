// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

typedef uint type_4[1];
struct RWStructuredBuffer {
    type_4 member;
};

void main_1(
    device RWStructuredBuffer& data,
    constant _mslBufferSizes& _buffer_sizes
) {
    uint phi_19_ = {};
    phi_19_ = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            phi_19_ = phi_19_ + 1u;
        }
        loop_init = false;
        uint _e7 = phi_19_;
        switch(0) {
            default: {
                if (_e7 < 4u) {
                } else {
                    break;
                }
                uint _e11 = data.member[_e7];
                if (_e11 == 1u) {
                    break;
                }
                uint _e13 = data.member[_e7];
                if (_e13 == 2u) {
                    break;
                }
                data.member[_e7] = _e7 * 2u;
                break;
            }
        }
        continue;
    }
    return;
}

kernel void main_(
  device RWStructuredBuffer& data [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    main_1(data, _buffer_sizes);
}
