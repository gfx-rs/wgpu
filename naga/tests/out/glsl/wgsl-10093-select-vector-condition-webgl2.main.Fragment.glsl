#version 300 es

precision highp float;
precision highp int;

flat in ivec4 _vs2fs_location0;
flat in ivec4 _vs2fs_location1;
layout(location = 0) out ivec4 _fs2p_location0;

void main() {
    ivec4 a = _vs2fs_location0;
    ivec4 b = _vs2fs_location1;
    ivec4 _e3 = (a * 2);
    ivec4 _e6 = (b + ivec4(7));
    bvec4 _e7 = lessThan(a, b);
    _fs2p_location0 = ivec4(_e7.x ? _e6.x : _e3.x, _e7.y ? _e6.y : _e3.y, _e7.z ? _e6.z : _e3.z, _e7.w ? _e6.w : _e3.w);
    return;
}

