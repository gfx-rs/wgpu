static const float4 v_f32_one = float4(1.0, 1.0, 1.0, 1.0);
static const float4 v_f32_zero = float4(0.0, 0.0, 0.0, 0.0);
static const float4 v_f32_half = float4(0.5, 0.5, 0.5, 0.5);
static const int4 v_i32_one = int4(int(1), int(1), int(1), int(1));
static const bool b_false = false;
static const bool b_true = true;
static const bool short_circuit_1_invalid_rhs = false;
static const bool short_circuit_2_invalid_rhs = false;
static const bool short_circuit_3_ = true;
static const bool short_circuit_4_ = true;

int4 naga_mod(int4 lhs, int4 rhs) {
    int4 divisor = ((lhs == int(-2147483647 - 1) & rhs == -1) | (rhs == 0)) ? 1 : rhs;
    return lhs - (lhs / divisor) * divisor;
}

int naga_div(int lhs, int rhs) {
    return lhs / (((lhs == int(-2147483647 - 1) & rhs == -1) | (rhs == 0)) ? 1 : rhs);
}

int naga_mod(int lhs, int rhs) {
    int divisor = ((lhs == int(-2147483647 - 1) & rhs == -1) | (rhs == 0)) ? 1 : rhs;
    return lhs - (lhs / divisor) * divisor;
}

bool ZeroValuebool() {
    return (bool)0;
}

int ZeroValueint() {
    return (int)0;
}

int3 ZeroValueint3() {
    return (int3)0;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_GroupID)
{
    float4 local = (float4)0;
    float m = (float)0;
    int n = (int)0;
    float4 local_1 = (float4)0;
    float2 a = (float2)0;
    float2 local_2 = (float2)0;
    float3 x = (float3)0;
    float3 local_3 = (float3)0;
    bool local_4 = (bool)0;
    bool local_5 = (bool)0;
    bool local_6 = (bool)0;
    bool local_7 = (bool)0;
    bool local_8 = (bool)0;
    bool local_9 = (bool)0;
    bool local_10 = (bool)0;
    bool local_11 = (bool)0;
    bool local_12 = (bool)0;
    bool local_13 = (bool)0;
    bool local_14 = (bool)0;
    bool local_15 = (bool)0;
    int prevent_const_eval = (int)0;
    int wgpu_7437_ = (int)0;
    int a_1 = (int)0;
    int3 vec0_ = (int3)0;

    local = (((((float4(asint(asuint(((true ? int(1) : int(0))).xxxx) + asuint(int4(int(0), int(0), int(0), int(0))))) + (true ? v_f32_one : v_f32_zero)) + lerp(v_f32_zero, v_f32_one, v_f32_half)) + lerp(v_f32_zero, v_f32_one, 0.1)) + (asfloat(int(1))).xxxx) + asfloat(v_i32_one));
    m = float(id.x);
    float _e39 = m;
    n = int(id.y);
    int _e41 = n;
    local_1 = (((((2.0).xx + (_e39).xx) - (4.0).xx) / (8.0).xx).xyxy + float4(naga_mod((_e41).xxxx, (int(2)).xxxx)));
    a = (2.0).xx;
    float2 _e62 = a;
    a = (_e62 + (1.0).xx);
    float2 _e66 = a;
    a = (_e66 - (3.0).xx);
    float2 _e70 = a;
    a = (_e70 / (4.0).xx);
    float2 _e74 = a;
    local_2 = _e74;
    x = float3(1.0, 1.0, 1.0);
    float3 _e80 = x;
    local_3 = float3(bool3(_e80));
    local_4 = ZeroValuebool();
    local_5 = ZeroValuebool();
    local_6 = ZeroValuebool();
    local_7 = ZeroValuebool();
    local_8 = ZeroValuebool();
    local_9 = ZeroValuebool();
    local_10 = ZeroValuebool();
    local_11 = ZeroValuebool();
    local_12 = ZeroValuebool();
    local_13 = ZeroValuebool();
    local_14 = ZeroValuebool();
    local_15 = ZeroValuebool();
    if (!(true)) {
        local_4 = false;
    } else {
        local_4 = true;
    }
    if (true) {
        local_5 = false;
    } else {
        local_5 = false;
    }
    if (!(false)) {
        local_6 = false;
    } else {
        local_6 = true;
    }
    local_11 = true;
    bool _e88 = local_11;
    if (!(_e88)) {
        local_12 = false;
        bool _e91 = local_12;
        local_7 = _e91;
    } else {
        local_7 = true;
    }
    bool _e93 = local_7;
    if (_e93) {
        local_13 = true;
        bool _e95 = local_13;
        if (!(_e95)) {
            local_14 = false;
            bool _e98 = local_14;
            local_9 = _e98;
        } else {
            local_9 = true;
        }
        bool _e100 = local_9;
        local_8 = _e100;
    } else {
        local_8 = false;
    }
    if (false) {
        local_15 = false;
        bool _e102 = local_15;
        local_10 = _e102;
    } else {
        local_10 = true;
    }
    prevent_const_eval = ZeroValueint();
    wgpu_7437_ = ZeroValueint();
    {
    }
    int _e104 = prevent_const_eval;
    wgpu_7437_ = asint(asuint(_e104) + asuint(int(-2147483647 - 1)));
    a_1 = ZeroValueint();
    vec0_ = ZeroValueint3();
    a_1 = int(1);
    int _e109 = a_1;
    a_1 = asint(asuint(_e109) + asuint(int(1)));
    int _e111 = a_1;
    a_1 = asint(asuint(_e111) - asuint(int(1)));
    int _e113 = a_1;
    int _e114 = a_1;
    a_1 = asint(asuint(_e113) * asuint(_e114));
    int _e116 = a_1;
    int _e117 = a_1;
    a_1 = naga_div(_e116, _e117);
    int _e119 = a_1;
    a_1 = naga_mod(_e119, int(1));
    int _e122 = a_1;
    a_1 = (_e122 & int(0));
    int _e124 = a_1;
    a_1 = (_e124 | int(0));
    int _e126 = a_1;
    a_1 = (_e126 ^ int(0));
    int _e129 = a_1;
    a_1 = (_e129 << 2u);
    int _e132 = a_1;
    a_1 = (_e132 >> 1u);
    int _e135 = a_1;
    a_1 = asint(asuint(_e135) + asuint(int(1)));
    int _e138 = a_1;
    a_1 = asint(asuint(_e138) - asuint(int(1)));
    int _e143 = vec0_[int(1)];
    vec0_[int(1)] = asint(asuint(_e143) + asuint(int(1)));
    int _e147 = vec0_[int(1)];
    vec0_[int(1)] = asint(asuint(_e147) - asuint(int(1)));
    return;
}
