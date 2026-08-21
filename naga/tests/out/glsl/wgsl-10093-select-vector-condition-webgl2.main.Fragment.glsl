#version 300 es

precision highp float;
precision highp int;

struct Inputs {
    ivec4 ia;
    ivec4 ib;
    uvec4 ua;
    uvec4 ub;
    vec4 fa;
    vec4 fb;
};
struct Outputs {
    ivec4 out_i;
    uvec4 out_u;
    vec4 out_f;
    ivec4 out_b;
};
flat in ivec4 _vs2fs_location0;
flat in ivec4 _vs2fs_location1;
flat in uvec4 _vs2fs_location2;
flat in uvec4 _vs2fs_location3;
smooth in vec4 _vs2fs_location4;
smooth in vec4 _vs2fs_location5;
layout(location = 0) out ivec4 _fs2p_location0;
layout(location = 1) out uvec4 _fs2p_location1;
layout(location = 2) out vec4 _fs2p_location2;
layout(location = 3) out ivec4 _fs2p_location3;

void main() {
    Inputs in_ = Inputs(_vs2fs_location0, _vs2fs_location1, _vs2fs_location2, _vs2fs_location3, _vs2fs_location4, _vs2fs_location5);
    Outputs out_ = Outputs(ivec4(0), uvec4(0u), vec4(0.0), ivec4(0));
    bvec4 icond = lessThan(in_.ia, in_.ib);
    out_.out_i = mix((in_.ia * 2), (in_.ib + ivec4(7)), icond);
    out_.out_u = mix(in_.ua, in_.ub, lessThan(in_.ua, in_.ub));
    out_.out_f = mix(in_.fa, in_.fb, lessThan(in_.fa, in_.fb));
    bvec4 picked = mix(icond, lessThan(in_.ua, in_.ub), icond);
    out_.out_b = mix(ivec4(0), ivec4(1), picked);
    Outputs _e38 = out_;
    _fs2p_location0 = _e38.out_i;
    _fs2p_location1 = _e38.out_u;
    _fs2p_location2 = _e38.out_f;
    _fs2p_location3 = _e38.out_b;
    return;
}

