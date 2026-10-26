#version 330 core
layout(location = 0) out vec4 _fs2p_location0;

ivec3 side_effect(int x) {
    return ivec3(x, (x + 1), (x + 2));
}

ivec3 u3_as_i3_(uvec3 u) {
    return ivec3(u);
}

void main() {
    vec4 p = gl_FragCoord;
    float acc = 0.0;
    bool local = false;
    bool local_1 = false;
    bvec2 b2_ = greaterThan(p.xy, vec2(0.5));
    bvec3 b3_ = greaterThan(p.xyz, vec3(0.5));
    bvec4 b4_ = greaterThan(p, vec4(0.5));
    ivec2 i2_ = ivec2(p.xy);
    ivec3 i3_ = ivec3(int(p.x), -2, 7);
    ivec4 i4_ = ivec4(p);
    uvec2 u2_ = uvec2(p.xy);
    uvec3 u3_ = uvec3(p.xyz);
    uvec4 u4_ = uvec4(p);
    vec2 f2_ = mix(p.xy, p.yx, b2_);
    vec3 f3_ = mix(p.xyz, p.zyx, b3_);
    vec4 f4_ = mix(p, p.wzyx, b4_);
    ivec2 _e33 = -(i2_);
    ivec2 si2_ = ivec2(b2_.x ? _e33.x : i2_.x, b2_.y ? _e33.y : i2_.y);
    ivec3 _e35 = -(i3_);
    bvec3 _e38 = lessThan(i3_, ivec3(0));
    ivec3 si3_ = ivec3(_e38.x ? _e35.x : i3_.x, _e38.y ? _e35.y : i3_.y, _e38.z ? _e35.z : i3_.z);
    ivec4 si4_ = ivec4(b4_.x ? i4_.x : ivec4(0).x, b4_.y ? i4_.y : ivec4(0).y, b4_.z ? i4_.z : ivec4(0).z, b4_.w ? i4_.w : ivec4(0).w);
    uvec2 _e44 = (u2_ + uvec2(1u));
    uvec2 su2_ = uvec2(b2_.x ? _e44.x : u2_.x, b2_.y ? _e44.y : u2_.y);
    uvec3 _e47 = (u3_ * 2u);
    uvec3 su3_ = uvec3(b3_.x ? _e47.x : u3_.x, b3_.y ? _e47.y : u3_.y, b3_.z ? _e47.z : u3_.z);
    uvec4 _e50 = uvec4(1u);
    uvec4 su4_ = uvec4(b4_.x ? u4_.x : _e50.x, b4_.y ? u4_.y : _e50.y, b4_.z ? u4_.z : _e50.z, b4_.w ? u4_.w : _e50.w);
    bvec2 _e52 = not(b2_);
    bvec2 _e55 = bvec2(true, false);
    bvec2 sb2_ = bvec2(_e55.x ? _e52.x : b2_.x, _e55.y ? _e52.y : b2_.y);
    bvec3 _e58 = bvec3(true);
    bvec3 _e61 = equal(b3_, bvec3(false));
    bvec3 sb3_ = bvec3(_e61.x ? _e58.x : b3_.x, _e61.y ? _e58.y : b3_.y, _e61.z ? _e58.z : b3_.z);
    bvec4 _e63 = not(b4_);
    bvec4 sb4_ = bvec4(b4_.x ? b4_.x : _e63.x, b4_.y ? b4_.y : _e63.y, b4_.z ? b4_.z : _e63.z, b4_.w ? b4_.w : _e63.w);
    ivec3 _e66 = side_effect(1);
    ivec3 _e68 = side_effect(2);
    ivec3 once = ivec3(b3_.x ? _e68.x : _e66.x, b3_.y ? _e68.y : _e66.y, b3_.z ? _e68.z : _e66.z);
    ivec3 _e70 = u3_as_i3_(u3_);
    ivec3 scalar = ((p.x > 0.5) ? _e70 : i3_);
    acc = ((f2_.x + f3_.y) + f4_.z);
    float _e81 = acc;
    acc = (_e81 + float(((((si2_.x + si3_.y) + si4_.z) + once.x) + scalar.y)));
    float _e93 = acc;
    acc = (_e93 + float(((su2_.x + su3_.y) + su4_.z)));
    float _e101 = acc;
    if (sb2_.x) {
        local = sb3_.y;
    } else {
        local = false;
    }
    bool _e107 = local;
    if (_e107) {
        local_1 = sb4_.z;
    } else {
        local_1 = false;
    }
    bool _e112 = local_1;
    acc = (_e101 + (_e112 ? 1.0 : 0.0));
    float _e117 = acc;
    _fs2p_location0 = vec4(_e117);
    return;
}

