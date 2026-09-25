// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    int inner[4];
};

int f(
    int x
) {
    return as_type<int>(as_type<uint>(x) * as_type<uint>(2));
}

void named(
) {
    int a = 0;
    uint2 loop_bound = uint2(4294967295u);
    bool loop_init = true;
    while(true) {
        if (metal::all(loop_bound == uint2(0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            if (a < 8) {
                break;
            }
        }
        loop_init = false;
        int _e2 = a;
        bool cmp = _e2 < 8;
        if (cmp) {
            int _e5 = a;
            a = as_type<int>(as_type<uint>(_e5) + as_type<uint>(1));
        }
    }
    return;
}

void call_result(
) {
    int a_1 = 0;
    uint2 loop_bound_1 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_1 == uint2(0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        int _e2 = a_1;
        int _e3 = f(_e2);
        int _e4 = a_1;
        a_1 = as_type<int>(as_type<uint>(_e4) + as_type<uint>(1));
    }
    return;
}

void through_pointer(
) {
    type_1 arr = type_1 {{0, 1, 2, 3}};
    int i = 0;
    uint2 loop_bound_2 = uint2(4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (metal::all(loop_bound_2 == uint2(0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_1) {
            int _e13 = arr.inner[i];
            if (arr.inner[i] > 2) {
                break;
            }
        }
        loop_init_1 = false;
        int _e8 = i;
        int _e10 = i;
        i = as_type<int>(as_type<uint>(_e10) + as_type<uint>(1));
    }
    return;
}

void nested(
) {
    int a_2 = 0;
    uint2 loop_bound_3 = uint2(4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (metal::all(loop_bound_3 == uint2(0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_2) {
            if (a_2 < 8) {
                break;
            }
        }
        loop_init_2 = false;
        int _e2 = a_2;
        bool outer = _e2 < 8;
        uint2 loop_bound_4 = uint2(4294967295u);
        bool loop_init_3 = true;
        while(true) {
            if (metal::all(loop_bound_4 == uint2(0u))) { break; }
            loop_bound_4 -= uint2(loop_bound_4.y == 0u, 1u);
            if (!loop_init_3) {
                if (a_2 < 4) {
                    break;
                }
            }
            loop_init_3 = false;
            int _e5 = a_2;
            bool inner = _e5 < 4;
            int _e8 = a_2;
            a_2 = as_type<int>(as_type<uint>(_e8) + as_type<uint>(1));
        }
    }
    return;
}

void dead_use(
) {
    int a_3 = 0;
    uint2 loop_bound_5 = uint2(4294967295u);
    bool loop_init_4 = true;
    while(true) {
        if (metal::all(loop_bound_5 == uint2(0u))) { break; }
        loop_bound_5 -= uint2(loop_bound_5.y == 0u, 1u);
        if (!loop_init_4) {
            int _e10 = a_3;
            if (_e10 > 5) {
            }
            int _e13 = a_3;
            if (a_3 > 10) {
                break;
            }
        }
        loop_init_4 = false;
        int _e2 = a_3;
        int _e3 = f(_e2);
        int _e4 = a_3;
        int w = as_type<int>(as_type<uint>(_e4) * as_type<uint>(2));
        int _e7 = a_3;
        a_3 = as_type<int>(as_type<uint>(_e7) + as_type<uint>(1));
    }
    return;
}

void restricted_image_load(
    metal::texture2d<float, metal::access::sample> tex
) {
    int i_1 = 0;
    uint2 loop_bound_6 = uint2(4294967295u);
    bool loop_init_5 = true;
    while(true) {
        if (metal::all(loop_bound_6 == uint2(0u))) { break; }
        loop_bound_6 -= uint2(loop_bound_6.y == 0u, 1u);
        if (!loop_init_5) {
            if (tex.read(metal::min(metal::uint2(metal::int2(i_1, 0)), metal::uint2(tex.get_width(clamped_lod_e7), tex.get_height(clamped_lod_e7)) - 1), clamped_lod_e7).x > 0.0) {
                break;
            }
        }
        loop_init_5 = false;
        int _e3 = i_1;
        uint clamped_lod_e7 = metal::min(uint(0), tex.get_num_mip_levels() - 1);
        metal::float4 _e7 = tex.read(metal::min(metal::uint2(metal::int2(_e3, 0)), metal::uint2(tex.get_width(clamped_lod_e7), tex.get_height(clamped_lod_e7)) - 1), clamped_lod_e7);
        float val = _e7.x;
        int _e9 = i_1;
        i_1 = as_type<int>(as_type<uint>(_e9) + as_type<uint>(1));
    }
    return;
}

uint one(
) {
    return 1u;
}

void call_result_as_index(
    device metal::float2 const& v0_
) {
    uint2 loop_bound_7 = uint2(4294967295u);
    while(true) {
        if (metal::all(loop_bound_7 == uint2(0u))) { break; }
        loop_bound_7 -= uint2(loop_bound_7.y == 0u, 1u);
        uint _e1 = one();
        float v1_ = v0_[_e1];
    }
    return;
}

void break_if_uses_continuing(
    thread int& counter
) {
    uint2 loop_bound_8 = uint2(4294967295u);
    bool loop_init_6 = true;
    while(true) {
        if (metal::all(loop_bound_8 == uint2(0u))) { break; }
        loop_bound_8 -= uint2(loop_bound_8.y == 0u, 1u);
        if (!loop_init_6) {
            int seen = counter;
            int _e3 = counter;
            counter = as_type<int>(as_type<uint>(_e3) + as_type<uint>(1));
            if (counter > 3) {
                break;
            }
        }
        loop_init_6 = false;
    }
    return;
}

kernel void main_(
  metal::texture2d<float, metal::access::sample> tex [[user(fake0)]]
, device metal::float2 const& v0_ [[user(fake0)]]
) {
    int counter = 0;
    named();
    call_result();
    through_pointer();
    nested();
    dead_use();
    restricted_image_load(tex);
    call_result_as_index(v0_);
    break_if_uses_continuing(counter);
    return;
}
