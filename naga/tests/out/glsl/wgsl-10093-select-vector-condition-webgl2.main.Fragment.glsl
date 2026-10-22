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
    ivec4 _e8 = (in_.ia * 2);
    ivec4 _e12 = (in_.ib + ivec4(7));
    out_.out_i = ivec4(icond.x ? _e12.x : _e8.x, icond.y ? _e12.y : _e8.y, icond.z ? _e12.z : _e8.z, icond.w ? _e12.w : _e8.w);
    uvec4 _e15 = in_.ua;
    uvec4 _e16 = in_.ub;
    bvec4 _e19 = lessThan(in_.ua, in_.ub);
    out_.out_u = uvec4(_e19.x ? _e16.x : _e15.x, _e19.y ? _e16.y : _e15.y, _e19.z ? _e16.z : _e15.z, _e19.w ? _e16.w : _e15.w);
    out_.out_f = mix(in_.fa, in_.fb, lessThan(in_.fa, in_.fb));
    bvec4 _e30 = lessThan(in_.ua, in_.ub);
    bvec4 picked = bvec4(icond.x ? _e30.x : icond.x, icond.y ? _e30.y : icond.y, icond.z ? _e30.z : icond.z, icond.w ? _e30.w : icond.w);
    ivec4 _e34 = ivec4(0);
    ivec4 _e36 = ivec4(1);
    out_.out_b = ivec4(picked.x ? _e36.x : _e34.x, picked.y ? _e36.y : _e34.y, picked.z ? _e36.z : _e34.z, picked.w ? _e36.w : _e34.w);
    Outputs _e38 = out_;
    _fs2p_location0 = _e38.out_i;
    _fs2p_location1 = _e38.out_u;
    _fs2p_location2 = _e38.out_f;
    _fs2p_location3 = _e38.out_b;
    return;
}

