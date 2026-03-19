// language: metal3.0
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

struct UniformIndex {
    uint index;
};
typedef int type_2[1];
struct Foo {
    uint x;
    type_2 far;
};
template <typename T>
struct NagaArgumentBufferWrapper {
    T inner;
};
struct FragmentIn {
    uint index;
};

struct main_Input {
    uint index [[user(loc0), flat]];
};
struct main_Output {
    uint member [[color(0)]];
};
fragment main_Output main_(
  main_Input varyings [[stage_in]]
, device NagaArgumentBufferWrapper<device Foo*>* storage_array [[buffer(0)]]
, constant UniformIndex& uni [[buffer(1)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    const FragmentIn fragment_in = { varyings.index };
    uint u1_ = 0u;
    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    uint _e7 = u1_;
    uint _e11 = storage_array[0].inner->x;
    u1_ = _e7 + _e11;
    uint _e13 = u1_;
    uint _e17 = uint(uniform_index) < 1 ? storage_array[uniform_index].inner->x : DefaultConstructible();
    u1_ = _e13 + _e17;
    uint _e19 = u1_;
    uint _e23 = uint(non_uniform_index) < 1 ? storage_array[non_uniform_index].inner->x : DefaultConstructible();
    u1_ = _e19 + _e23;
    uint _e25 = u1_;
    u1_ = _e25 + (1 + (_buffer_sizes.size0 - 4 - 4) / 4);
    uint _e31 = u1_;
    u1_ = _e31 + (1 + (_buffer_sizes.size0 - 4 - 4) / 4);
    uint _e37 = u1_;
    u1_ = _e37 + (1 + (_buffer_sizes.size0 - 4 - 4) / 4);
    uint _e43 = u1_;
    return main_Output { _e43 };
}
