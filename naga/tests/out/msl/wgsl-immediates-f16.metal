// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size1;
};

struct Inner {
    half h;
    char _pad1[2];
    int i;
};
struct Data {
    half a;
    half b;
    char _pad2[4];
    metal::half3 v3_;
    metal::half2x3 m;
    Inner inner;
    float f;
    uint u;
};
typedef float type_6[1];

struct main_Input {
};
kernel void main_(
  uint i [[thread_index_in_threadgroup]]
, constant Data& data [[user(fake0)]]
, device type_6& out [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    Data whole = data;
    out[0] = static_cast<float>(whole.a);
    half _e11 = data.b;
    out[1] = static_cast<float>(_e11);
    half _e18 = data.v3_.z;
    out[2] = static_cast<float>(_e18);
    half _e25 = data.v3_[i];
    out[3] = static_cast<float>(_e25);
    metal::half3 column = data.m[1];
    out[4] = static_cast<float>(column.y);
    half _e41 = data.m[i].z;
    out[5] = static_cast<float>(_e41);
    half _e48 = data.inner.h;
    int _e53 = data.inner.i;
    out[6] = static_cast<float>(_e48) + static_cast<float>(_e53);
    float _e60 = data.f;
    uint _e63 = data.u;
    out[7] = _e60 + static_cast<float>(_e63);
    return;
}
