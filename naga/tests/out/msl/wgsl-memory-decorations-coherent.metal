// language: metal3.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
    uint size1;
};

typedef uint type_1[1];
struct Data {
    type_1 values;
};

kernel void main_(
  coherent device Data& coherent_buf [[user(fake0)]]
, device Data const& plain_buf [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    uint _e6 = plain_buf.values[0];
    coherent_buf.values[0] = _e6;
    return;
}
