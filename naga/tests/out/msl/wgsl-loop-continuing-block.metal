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
    metal::uint2 loop_bound = metal::uint2(4294967295u);
    bool cmp = {};
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            if (cmp) {
                break;
            }
        }
        loop_init = false;
        {
            metal::uint2 _e13 = loop_bound;
            loop_bound = _e13 - metal::uint2((_e13.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e13 == metal::uint2(0u))) {
                break;
            }
        }
        int _e2 = a;
        cmp = _e2 < 8;
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
    metal::uint2 loop_bound_1 = metal::uint2(4294967295u);
    int v = {};
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            if (v > 10) {
                break;
            }
        }
        loop_init_1 = false;
        {
            metal::uint2 _e14 = loop_bound_1;
            loop_bound_1 = _e14 - metal::uint2((_e14.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e14 == metal::uint2(0u))) {
                break;
            }
        }
        int _e2 = a_1;
        v = f(_e2);
        int _e4 = a_1;
        a_1 = as_type<int>(as_type<uint>(_e4) + as_type<uint>(1));
    }
    return;
}

void through_pointer(
) {
    type_1 arr = type_1 {{0, 1, 2, 3}};
    int i = 0;
    metal::uint2 loop_bound_2 = metal::uint2(4294967295u);
    int _e8 = {};
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            int _e13 = arr.inner[_e8];
            if (_e13 > 2) {
                break;
            }
        }
        loop_init_2 = false;
        {
            metal::uint2 _e21 = loop_bound_2;
            loop_bound_2 = _e21 - metal::uint2((_e21.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e21 == metal::uint2(0u))) {
                break;
            }
        }
        _e8 = i;
        int _e10 = i;
        i = as_type<int>(as_type<uint>(_e10) + as_type<uint>(1));
    }
    return;
}

void nested(
) {
    int a_2 = 0;
    metal::uint2 loop_bound_3 = metal::uint2(4294967295u);
    metal::uint2 loop_bound_4 = metal::uint2(4294967295u);
    bool outer = {};
    bool loop_init_3 = true;
    while(true) {
        if (!loop_init_3) {
            if (outer) {
                break;
            }
        }
        loop_init_3 = false;
        {
            metal::uint2 _e31 = loop_bound_3;
            loop_bound_3 = _e31 - metal::uint2((_e31.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e31 == metal::uint2(0u))) {
                break;
            }
        }
        int _e2 = a_2;
        outer = _e2 < 8;
        loop_bound_4 = metal::uint2(4294967295u);
        bool inner = {};
        bool loop_init_4 = true;
        while(true) {
            if (!loop_init_4) {
                if (inner) {
                    break;
                }
            }
            loop_init_4 = false;
            {
                metal::uint2 _e19 = loop_bound_4;
                loop_bound_4 = _e19 - metal::uint2((_e19.y == 0u) ? 1u : 0u, 1u);
                if (metal::all(_e19 == metal::uint2(0u))) {
                    break;
                }
            }
            int _e5 = a_2;
            inner = _e5 < 4;
            int _e8 = a_2;
            a_2 = as_type<int>(as_type<uint>(_e8) + as_type<uint>(1));
        }
    }
    return;
}

void dead_use(
) {
    int a_3 = 0;
    metal::uint2 loop_bound_5 = metal::uint2(4294967295u);
    int v_1 = {};
    int w = {};
    bool loop_init_5 = true;
    while(true) {
        if (!loop_init_5) {
            int _e10 = a_3;
            if (_e10 > 5) {
                int dead = as_type<int>(as_type<uint>(v_1) + as_type<uint>(w));
            }
            int _e14 = a_3;
            if (_e14 > 10) {
                break;
            }
        }
        loop_init_5 = false;
        {
            metal::uint2 _e22 = loop_bound_5;
            loop_bound_5 = _e22 - metal::uint2((_e22.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e22 == metal::uint2(0u))) {
                break;
            }
        }
        int _e2 = a_3;
        v_1 = f(_e2);
        int _e4 = a_3;
        w = as_type<int>(as_type<uint>(_e4) * as_type<uint>(2));
        int _e7 = a_3;
        a_3 = as_type<int>(as_type<uint>(_e7) + as_type<uint>(1));
    }
    return;
}

void restricted_image_load(
    metal::texture2d<float, metal::access::sample> tex
) {
    int i_1 = 0;
    metal::uint2 loop_bound_6 = metal::uint2(4294967295u);
    float val = {};
    bool loop_init_6 = true;
    while(true) {
        if (!loop_init_6) {
            if (val > 0.0) {
                break;
            }
        }
        loop_init_6 = false;
        {
            metal::uint2 _e19 = loop_bound_6;
            loop_bound_6 = _e19 - metal::uint2((_e19.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e19 == metal::uint2(0u))) {
                break;
            }
        }
        int _e3 = i_1;
        uint clamped_lod_e7 = metal::min(uint(0), tex.get_num_mip_levels() - 1);
        metal::float4 _e7 = tex.read(metal::min(metal::uint2(metal::int2(_e3, 0)), metal::uint2(tex.get_width(clamped_lod_e7), tex.get_height(clamped_lod_e7)) - 1), clamped_lod_e7);
        val = _e7.x;
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
    metal::uint2 loop_bound_7 = metal::uint2(4294967295u);
    float v1_ = {};
    bool loop_init_7 = true;
    while(true) {
        if (!loop_init_7) {
            if (v1_ > 0.0) {
                break;
            }
        }
        loop_init_7 = false;
        {
            metal::uint2 _e11 = loop_bound_7;
            loop_bound_7 = _e11 - metal::uint2((_e11.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e11 == metal::uint2(0u))) {
                break;
            }
        }
        uint _e1 = one();
        v1_ = v0_[_e1];
    }
    return;
}

void break_if_uses_continuing(
    thread int& counter
) {
    metal::uint2 loop_bound_8 = metal::uint2(4294967295u);
    bool loop_init_8 = true;
    while(true) {
        if (!loop_init_8) {
            int seen = counter;
            int _e3 = counter;
            counter = as_type<int>(as_type<uint>(_e3) + as_type<uint>(1));
            if (seen > 3) {
                break;
            }
        }
        loop_init_8 = false;
        {
            metal::uint2 _e13 = loop_bound_8;
            loop_bound_8 = _e13 - metal::uint2((_e13.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e13 == metal::uint2(0u))) {
                break;
            }
        }
    }
    return;
}

void name_reuse(
) {
    float a_4 = 0.0;
    int a_5 = {};
    float b = {};
    metal::uint2 loop_bound_9 = metal::uint2(4294967295u);
    bool loop_init_9 = true;
    while(true) {
        if (!loop_init_9) {
            int _e7 = a_5;
            if (_e7 == 2) {
                break;
            }
        }
        loop_init_9 = false;
        {
            metal::uint2 _e18 = loop_bound_9;
            loop_bound_9 = _e18 - metal::uint2((_e18.y == 0u) ? 1u : 0u, 1u);
            if (metal::all(_e18 == metal::uint2(0u))) {
                break;
            }
        }
        a_5 = 1;
        int _e4 = a_5;
        a_5 = as_type<int>(as_type<uint>(_e4) + as_type<uint>(1));
    }
    float _e10 = a_4;
    b = metal::cos(_e10);
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
    name_reuse();
    return;
}
