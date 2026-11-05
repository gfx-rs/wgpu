// language: metal2.1
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;
struct DefaultConstructible {
    template<typename T>
    operator T() && {
        return T {};
    }
};

struct _mslBufferSizes {
    uint size0;
};

typedef uint type_1[1];

void store(
    uint index_1,
    device uint* values,
    constant _mslBufferSizes& _buffer_sizes
) {
    if (uint(index_1) < 1 + (_buffer_sizes.size0 - 0 - 4) / 4) {
        values[index_1] = index_1 + 1u;
    }
    return;
}

struct main_Input {
};
kernel void main_(
  uint index [[thread_index_in_threadgroup]]
, device uint* values [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    store(index, values, _buffer_sizes);
    return;
}
