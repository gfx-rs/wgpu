#version 400 core
layout(location = 0) out vec4 _fs2p_location0;

int test_f32_to_i32_(float f) {
    return int(clamp(f, -2147483648.0, 2147483520.0));
}

uint test_f32_to_u32_(float f_1) {
    return uint(clamp(f_1, 0.0, 4294967040.0));
}

int test_f64_to_i32_(double f_2) {
    return int(clamp(f_2, -2147483648.0LF, 2147483647.0LF));
}

uint test_f64_to_u32_(double f_3) {
    return uint(clamp(f_3, 0.0LF, 4294967295.0LF));
}

ivec2 test_f32_to_i32_vec(vec2 f_4) {
    return ivec2(clamp(f_4, -2147483648.0, 2147483520.0));
}

uvec2 test_f32_to_u32_vec(vec2 f_5) {
    return uvec2(clamp(f_5, 0.0, 4294967040.0));
}

ivec2 test_f64_to_i32_vec(dvec2 f_6) {
    return ivec2(clamp(f_6, -2147483648.0LF, 2147483647.0LF));
}

uvec2 test_f64_to_u32_vec(dvec2 f_7) {
    return uvec2(clamp(f_7, 0.0LF, 4294967295.0LF));
}

uint test_u32_to_f32_to_u32_(uint x) {
    return uint(clamp(float(x), 0.0, 4294967040.0));
}

vec4 test_unaffected(int i, uint u, float f_8) {
    return vec4(float(i), float(u), intBitsToFloat(i), float(floatBitsToInt(f_8)));
}

void main() {
    vec4 pos = gl_FragCoord;
    int _e2 = test_f32_to_i32_(pos.x);
    int _e5 = test_f64_to_i32_(double(pos.x));
    ivec2 _e8 = test_f32_to_i32_vec(pos.xy);
    ivec2 _e13 = test_f64_to_i32_vec(dvec2(pos.xy));
    int i_1 = (((_e2 + _e5) + _e8.y) + _e13.y);
    uint _e17 = test_f32_to_u32_(pos.y);
    uint _e20 = test_f64_to_u32_(double(pos.y));
    uvec2 _e23 = test_f32_to_u32_vec(pos.xy);
    uvec2 _e28 = test_f64_to_u32_vec(dvec2(pos.xy));
    uint _e33 = test_u32_to_f32_to_u32_(uint(clamp(pos.z, 0.0, 4294967040.0)));
    uint u_1 = ((((_e17 + _e20) + _e23.x) + _e28.x) + _e33);
    vec4 _e36 = test_unaffected(i_1, u_1, pos.w);
    _fs2p_location0 = _e36;
    return;
}

