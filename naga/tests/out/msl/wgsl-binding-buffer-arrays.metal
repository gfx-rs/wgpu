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
struct Inner {
    uint y;
};
typedef int type_2[1];
struct Foo {
    uint x;
    Inner nested;
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
    uint _e30 = storage_array[0].inner->nested.y;
    u1_ = _e25 + _e30;
    uint _e32 = u1_;
    uint _e37 = uint(uniform_index) < 1 ? storage_array[uniform_index].inner->nested.y : DefaultConstructible();
    u1_ = _e32 + _e37;
    uint _e39 = u1_;
    uint _e44 = uint(non_uniform_index) < 1 ? storage_array[non_uniform_index].inner->nested.y : DefaultConstructible();
    u1_ = _e39 + _e44;
    uint _e46 = u1_;
    u1_ = _e46 + (1 + (_buffer_sizes.size0 - 8 - 4) / 4);
    uint _e52 = u1_;
    u1_ = _e52 + (1 + (_buffer_sizes.size0 - 8 - 4) / 4);
    uint _e58 = u1_;
    u1_ = _e58 + (1 + (_buffer_sizes.size0 - 8 - 4) / 4);
    uint _e64 = u1_;
    return main_Output { _e64 };
}
