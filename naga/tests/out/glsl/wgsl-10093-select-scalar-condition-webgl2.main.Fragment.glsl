#version 300 es

precision highp float;
precision highp int;

struct Inputs {
    ivec3 a;
    ivec3 b;
    int choice;
};
flat in ivec3 _vs2fs_location0;
flat in ivec3 _vs2fs_location1;
flat in int _vs2fs_location2;
layout(location = 0) out ivec3 _fs2p_location0;

void main() {
    Inputs in_ = Inputs(_vs2fs_location0, _vs2fs_location1, _vs2fs_location2);
    _fs2p_location0 = ((in_.choice != 0) ? in_.b : in_.a);
    return;
}

