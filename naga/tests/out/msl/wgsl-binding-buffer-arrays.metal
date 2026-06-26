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
    uint size0[10];
    uint size1;
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
typedef uint type_3[1];
struct PlainData {
    type_3 values;
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
, device PlainData const& plain_storage [[buffer(1)]]
, constant UniformIndex& uni [[buffer(2)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    const FragmentIn fragment_in = { varyings.index };
    uint u1_ = 0u;
    uint uniform_index = uni.index;
    uint non_uniform_index = fragment_in.index;
    uint _e7 = u1_;
    uint _e11 = uint(0) < 10 ? storage_array[0].inner->x : DefaultConstructible();
    u1_ = _e7 + _e11;
    uint _e13 = u1_;
    uint _e17 = uint(uniform_index) < 10 ? storage_array[uniform_index].inner->x : DefaultConstructible();
    u1_ = _e13 + _e17;
    uint _e19 = u1_;
    uint _e23 = uint(non_uniform_index) < 10 ? storage_array[non_uniform_index].inner->x : DefaultConstructible();
    u1_ = _e19 + _e23;
    uint _e25 = u1_;
    uint _e30 = uint(0) < 10 ? storage_array[0].inner->nested.y : DefaultConstructible();
    u1_ = _e25 + _e30;
    uint _e32 = u1_;
    uint _e37 = uint(uniform_index) < 10 ? storage_array[uniform_index].inner->nested.y : DefaultConstructible();
    u1_ = _e32 + _e37;
    uint _e39 = u1_;
    uint _e44 = uint(non_uniform_index) < 10 ? storage_array[non_uniform_index].inner->nested.y : DefaultConstructible();
    u1_ = _e39 + _e44;
    uint _e46 = u1_;
    u1_ = _e46 + (1 + (_buffer_sizes.size0[0u] - 8 - 4) / 4);
    uint _e52 = u1_;
    u1_ = _e52 + (1 + (_buffer_sizes.size0[unsigned(uniform_index)] - 8 - 4) / 4);
    uint _e58 = u1_;
    u1_ = _e58 + (1 + (_buffer_sizes.size0[unsigned(non_uniform_index)] - 8 - 4) / 4);
    uint _e64 = u1_;
    uint _e68 = uint(0) < 1 + (_buffer_sizes.size1 - 0 - 4) / 4 ? plain_storage.values[0] : DefaultConstructible();
    u1_ = _e64 + _e68;
    uint _e70 = u1_;
    u1_ = _e70 + (1 + (_buffer_sizes.size1 - 0 - 4) / 4);
    uint _e75 = u1_;
    return main_Output { _e75 };
}
