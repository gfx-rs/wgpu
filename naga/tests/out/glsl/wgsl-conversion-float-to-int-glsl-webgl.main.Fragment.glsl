#version 300 es

precision highp float;
precision highp int;

layout(location = 0) out vec4 _fs2p_location0;

int test_f32_to_i32_(float f) {
    return int(clamp(f, -2147483648.0, 2147483520.0));
}

uint test_f32_to_u32_(float f_1) {
    return uint(clamp(f_1, 0.0, 4294967040.0));
}

ivec2 test_f32_to_i32_vec(vec2 f_2) {
    return ivec2(clamp(f_2, -2147483648.0, 2147483520.0));
}

uvec2 test_f32_to_u32_vec(vec2 f_3) {
    return uvec2(clamp(f_3, 0.0, 4294967040.0));
}

uint test_u32_to_f32_to_u32_(uint x) {
    return uint(clamp(float(x), 0.0, 4294967040.0));
}

vec4 test_unaffected(int i, uint u, float f_4) {
    return vec4(float(i), float(u), intBitsToFloat(i), float(floatBitsToInt(f_4)));
}

void main() {
    vec4 pos = gl_FragCoord;
    int _e2 = test_f32_to_i32_(pos.x);
    ivec2 _e4 = test_f32_to_i32_vec(pos.xy);
    int i_1 = (_e2 + _e4.y);
    uint _e8 = test_f32_to_u32_(pos.y);
    uvec2 _e10 = test_f32_to_u32_vec(pos.xy);
    uint _e15 = test_u32_to_f32_to_u32_(uint(clamp(pos.z, 0.0, 4294967040.0)));
    uint u_1 = ((_e8 + _e10.x) + _e15);
    vec4 _e18 = test_unaffected(i_1, u_1, pos.w);
    _fs2p_location0 = _e18;
    return;
}

