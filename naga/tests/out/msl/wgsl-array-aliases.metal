// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size0;
};

struct A {
    uint inner[5];
};
typedef A B;
struct Left {
    uint value;
};
struct Right {
    uint value;
};
typedef uint type_1[1];
typedef A type_2;
struct type_3 {
    uint inner[4];
};
struct type_5 {
    int inner[5];
};
struct type_6 {
    Left inner[5];
};
struct type_7 {
    Right inner[5];
};

uint first(
    B indices
) {
    return indices.inner[0];
}

B return_array(
    A indices_1
) {
    return indices_1;
}

kernel void main_(
  device type_1& data [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    A a = {};
    B b = {};
    type_2 unnamed = {};
    type_3 shorter = {};
    type_5 signed_ = {};
    type_6 left = {};
    type_7 right = {};
    uint _e4 = data[0];
    a.inner[0] = _e4;
    uint _e8 = data[1];
    a.inner[1] = _e8;
    uint _e12 = data[2];
    a.inner[2] = _e12;
    uint _e16 = data[3];
    a.inner[3] = _e16;
    uint _e20 = data[4];
    a.inner[4] = _e20;
    A _e21 = a;
    b = _e21;
    B _e23 = b;
    B _e24 = return_array(_e23);
    unnamed = _e24;
    type_2 _e26 = unnamed;
    b = _e26;
    A _e33 = a;
    uint _e34 = first(_e33);
    uint _e36 = b.inner[1];
    uint _e39 = shorter.inner[0];
    int _e42 = signed_.inner[0];
    uint _e47 = left.inner[0].value;
    uint _e51 = right.inner[0].value;
    data[5] = ((((_e34 + _e36) + _e39) + static_cast<uint>(_e42)) + _e47) + _e51;
    return;
}
