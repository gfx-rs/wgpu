// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint size1;
    uint size2;
};

struct type_1 {
    uint inner[4];
};
struct S {
    uint a;
    type_1 b;
};
typedef uint type_2[1];
typedef float type_4[1];
struct type_7 {
    uint inner[64];
};
struct type_14 {
    uint inner[2];
};

void store(
    device uint& p,
    uint v
) {
    p = v;
    return;
}

uint load_add(
    device uint& p_1,
    uint v_1
) {
    uint old = p_1;
    p_1 = old + v_1;
    return old;
}

int find(
    threadgroup type_7& p_2,
    uint needle
) {
    int i_2 = 0;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            int _e13 = i_2;
            i_2 = as_type<int>(as_type<uint>(_e13) + as_type<uint>(1));
        }
        loop_init = false;
        int _e4 = i_2;
        if (_e4 < 64) {
        } else {
            break;
        }
        {
            int _e7 = i_2;
            uint _e9 = p_2.inner[_e7];
            if (_e9 == needle) {
                int _e11 = i_2;
                return _e11;
            }
        }
    }
    return -1;
}

uint classify(
    device uint& p_3
) {
    uint _e1 = p_3;
    switch(_e1) {
        case 0u: {
            return 10u;
        }
        case 1u: {
            p_3 = 2u;
            break;
        }
        default: {
            return 30u;
        }
    }
    return 20u;
}

void bump(
    thread uint& p_4
) {
    uint count = 1u;
    uint _e3 = p_4;
    uint _e4 = count;
    p_4 = _e3 + _e4;
    uint _e6 = count;
    count = _e6 + 1u;
    return;
}

void bump_twice(
    thread uint& p_5
) {
    bump(p_5);
    bump(p_5);
    return;
}

void set_private(
    thread uint& p_6
) {
    p_6 = 7u;
    return;
}

void early(
    device uint& p_7,
    bool c
) {
    if (c) {
        return;
    }
    p_7 = 5u;
    return;
}

uint not_inlined(
    thread uint& p_8
) {
    uint _e1 = p_8;
    return _e1;
}

struct main_Input {
};
kernel void main_(
  uint index [[thread_index_in_threadgroup]]
, device S& buf [[user(fake0)]]
, device type_2& arr [[user(fake0)]]
, threadgroup type_7& wg
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    type_1 private_arr = {};
    if (index == 0u) {
        wg = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    type_14 local = type_14 {};
    uint i = 0u;
    uint whole = 3u;
    store(buf.a, index);
    store(arr[index], 1u);
    uint _e10 = load_add(buf.b.inner[1], 2u);
    uint _e15 = load_add(buf.b.inner[2], 3u);
    uint x = _e10 + _e15;
    arr[0] = x;
    int _e22 = find(wg, index);
    arr[1] = static_cast<uint>(_e22);
    uint _e28 = classify(arr[3]);
    arr[2] = _e28;
    uint2 loop_bound_1 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e39 = i;
            i = _e39 + 1u;
        }
        loop_init_1 = false;
        uint _e33 = i;
        if (_e33 < 2u) {
        } else {
            break;
        }
        {
            uint _e36 = i;
            bump_twice(local.inner[_e36]);
        }
    }
    set_private(private_arr.inner[1]);
    early(arr[4], index == 0u);
    uint _e52 = local.inner[0];
    uint _e55 = private_arr.inner[1];
    uint _e57 = not_inlined(whole);
    arr[5] = (_e52 + _e55) + _e57;
    return;
}


struct fragInput {
    metal::float2 uv [[user(loc0), center_perspective]];
};
struct fragOutput {
    metal::float4 member_1 [[color(0)]];
};
fragment fragOutput frag(
  fragInput varyings_1 [[stage_in]]
, device type_4 const& lut [[user(fake0)]]
, metal::texture2d<float, metal::access::sample> t [[user(fake0)]]
, metal::sampler s [[user(fake0)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    const auto uv = varyings_1.uv;
    uint i_1 = {};
    float local_1 = {};
    bool local_2 = {};
    switch(0) {
        default: {
            i_1 = 0u;
            local_2 = false;
            uint2 loop_bound_2 = uint2(4294967295u);
            bool loop_init_2 = true;
            while(true) {
                if (metal::all(loop_bound_2 == uint2(0u))) { break; }
                loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
                if (!loop_init_2) {
                    uint _e14 = i_1;
                    i_1 = _e14 + 1u;
                }
                loop_init_2 = false;
                uint _e2 = i_1;
                if (_e2 < (1 + (_buffer_sizes.size2 - 0 - 4) / 4)) {
                } else {
                    break;
                }
                {
                    uint _e5 = i_1;
                    float _e7 = lut[_e5];
                    if (_e7 > 0.0) {
                        uint _e10 = i_1;
                        float _e12 = lut[_e10];
                        local_1 = _e12;
                        local_2 = true;
                        break;
                    }
                }
            }
            bool _e17 = local_2;
            if (_e17) {
                break;
            }
            local_1 = 0.0;
            break;
        }
    }
    float v_2 = local_1;
    metal::float4 _e23 = t.sample(s, uv);
    return fragOutput { _e23 * v_2 };
}
