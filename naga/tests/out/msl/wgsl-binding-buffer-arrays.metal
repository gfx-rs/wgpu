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
    uint _e11 = uint(0) < 10 && _buffer_sizes.size0[0u] != 0u ? storage_array[0].inner->x : DefaultConstructible();
    u1_ = _e7 + _e11;
    uint _e13 = u1_;
    uint _e17 = uint(uniform_index) < 10 && _buffer_sizes.size0[unsigned(uniform_index)] != 0u ? storage_array[uniform_index].inner->x : DefaultConstructible();
    u1_ = _e13 + _e17;
    uint _e19 = u1_;
    uint _e23 = uint(non_uniform_index) < 10 && _buffer_sizes.size0[unsigned(non_uniform_index)] != 0u ? storage_array[non_uniform_index].inner->x : DefaultConstructible();
    u1_ = _e19 + _e23;
    uint _e25 = u1_;
    uint _e29 = uint(7) < 10 && _buffer_sizes.size0[7u] != 0u ? storage_array[7].inner->x : DefaultConstructible();
    u1_ = _e25 + _e29;
    uint _e31 = u1_;
    uint _e36 = uint(0) < 10 && _buffer_sizes.size0[0u] != 0u ? storage_array[0].inner->nested.y : DefaultConstructible();
    u1_ = _e31 + _e36;
    uint _e38 = u1_;
    uint _e43 = uint(uniform_index) < 10 && _buffer_sizes.size0[unsigned(uniform_index)] != 0u ? storage_array[uniform_index].inner->nested.y : DefaultConstructible();
    u1_ = _e38 + _e43;
    uint _e45 = u1_;
    uint _e50 = uint(non_uniform_index) < 10 && _buffer_sizes.size0[unsigned(non_uniform_index)] != 0u ? storage_array[non_uniform_index].inner->nested.y : DefaultConstructible();
    u1_ = _e45 + _e50;
    uint _e52 = u1_;
    uint _e57 = uint(7) < 10 && _buffer_sizes.size0[7u] != 0u ? storage_array[7].inner->nested.y : DefaultConstructible();
    u1_ = _e52 + _e57;
    uint _e59 = u1_;
    u1_ = _e59 + (1 + (_buffer_sizes.size0[0u] - 8 - 4) / 4);
    uint _e65 = u1_;
    u1_ = _e65 + (1 + (_buffer_sizes.size0[unsigned(uniform_index)] - 8 - 4) / 4);
    uint _e71 = u1_;
    u1_ = _e71 + (1 + (_buffer_sizes.size0[unsigned(non_uniform_index)] - 8 - 4) / 4);
    uint _e77 = u1_;
    u1_ = _e77 + (1 + (_buffer_sizes.size0[7u] - 8 - 4) / 4);
    uint _e83 = u1_;
    int _e88 = uint(0) < 10 && _buffer_sizes.size0[0u] != 0u && uint(0) < 10 && _buffer_sizes.size0[0u] != 0u ? storage_array[0].inner->far[0] : DefaultConstructible();
    u1_ = _e83 + as_type<uint>(_e88);
    uint _e91 = u1_;
    int _e96 = uint(0) < 10 && _buffer_sizes.size0[0u] != 0u && uint(uniform_index) < 10 && _buffer_sizes.size0[unsigned(uniform_index)] != 0u ? storage_array[uniform_index].inner->far[0] : DefaultConstructible();
    u1_ = _e91 + as_type<uint>(_e96);
    uint _e99 = u1_;
    int _e104 = uint(0) < 10 && _buffer_sizes.size0[0u] != 0u && uint(non_uniform_index) < 10 && _buffer_sizes.size0[unsigned(non_uniform_index)] != 0u ? storage_array[non_uniform_index].inner->far[0] : DefaultConstructible();
    u1_ = _e99 + as_type<uint>(_e104);
    uint _e107 = u1_;
    int _e112 = uint(0) < 10 && _buffer_sizes.size0[0u] != 0u && uint(7) < 10 && _buffer_sizes.size0[7u] != 0u ? storage_array[7].inner->far[0] : DefaultConstructible();
    u1_ = _e107 + as_type<uint>(_e112);
    uint _e115 = u1_;
    uint _e119 = uint(0) < 1 + (_buffer_sizes.size1 - 0 - 4) / 4 ? plain_storage.values[0] : DefaultConstructible();
    u1_ = _e115 + _e119;
    uint _e121 = u1_;
    u1_ = _e121 + (1 + (_buffer_sizes.size1 - 0 - 4) / 4);
    uint _e126 = u1_;
    return main_Output { _e126 };
}
