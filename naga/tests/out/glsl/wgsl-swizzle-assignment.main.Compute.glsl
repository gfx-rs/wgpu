#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uvec4 v_1[1] = uvec4[1](uvec4(0u));

layout(std430) buffer type_7_block_0Compute { vec3 _group_0_binding_0_cs[]; };

shared mat2x4 w;


uint update_and_index() {
    uvec4 _e2 = v_1[0];
    v_1[0] = (_e2 + uvec4(2u, 4u, 8u, 10u));
    return 0u;
}

uvec2 update_and_rhs() {
    v_1[0].x = 900u;
    v_1[0].z = 100u;
    return uvec2(55u, 66u);
}

uvec4 eval_order() {
    v_1[0] = uvec4(0u, 0u, 0u, 0u);
    uint _e8 = update_and_index();
    uvec2 _e10 = update_and_rhs();
    uvec4 _e11 = v_1[_e8];
    v_1[_e8] = uvec4(_e10.y, _e10.x, _e11.z, _e11.w);
    uvec4 assigned = v_1[0];
    v_1[0] = uvec4(0u, 0u, 0u, 0u);
    uint _e28 = update_and_index();
    uvec4 _e30 = v_1[_e28];
    uvec2 _e32 = update_and_rhs();
    uvec2 _e33 = (_e30.yx + _e32);
    uvec4 _e34 = v_1[_e28];
    v_1[_e28] = uvec4(_e33.y, _e33.x, _e34.z, _e34.w);
    uvec4 _e42 = v_1[0];
    return (assigned + _e42);
}

uvec4 swizzle_views(int i) {
    uvec4 v = uvec4(0u);
    uvec4 _e4 = v;
    v = uvec4(_e4.x, 1u, 2u, _e4.w);
    uvec4 _e8 = v;
    uvec3 u = _e8.wyz;
    v = uvec4(1u, 2u, 3u, 4u);
    v = uvec4(4u, 3u, 2u, 1u);
    uvec4 _e22 = v;
    v = uvec4(_e22.x, 100u, 99u, _e22.w);
    v.z = 99u;
    uint _e30 = v.z;
    v.z = (_e30 + 1u);
    v.y = 50u;
    v[((i == 1) ? 0 : 3)] = 7u;
    v[i] = 8u;
    uvec4 _e43 = v;
    uvec2 _e46 = (_e43.xz * 3u);
    uvec4 _e47 = v;
    v = uvec4(_e46.x, _e47.y, _e46.y, _e47.w);
    uvec4 _e53 = v;
    uvec2 _e58 = (_e53.yw << uvec2(1u, 2u));
    uvec4 _e59 = v;
    v = uvec4(_e59.x, _e58.x, _e59.z, _e58.y);
    uvec4 _e65 = v;
    return (_e65 + u.xyzx);
}

void via_pointer(inout uvec4 p) {
    uvec4 _e3 = p;
    p = uvec4(_e3.x, 1u, 2u, _e3.w);
    uvec4 _e7 = p;
    uvec3 u_1 = _e7.wyz;
    uvec4 _e11 = p;
    p = uvec4(_e11.x, 100u, 99u, _e11.w);
    p.y = 50u;
    uvec2 _e17 = u_1.xy;
    uvec4 _e18 = p;
    p = uvec4(_e17.x, _e18.y, _e18.z, _e17.y);
    return;
}

void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        w = mat2x4(0.0);
    }
    memoryBarrierShared();
    barrier();
    uint index = gl_LocalInvocationIndex;
    uvec4 a = uvec4(0u);
    via_pointer(a);
    vec3 _e7 = _group_0_binding_0_cs[index];
    _group_0_binding_0_cs[index] = vec3(2.0, _e7.y, 1.0);
    vec3 _e12 = _group_0_binding_0_cs[index];
    vec2 _e16 = (_e12.yz - vec2(1.0));
    vec3 _e17 = _group_0_binding_0_cs[index];
    _group_0_binding_0_cs[index] = vec3(_e17.x, _e16.x, _e16.y);
    vec4 _e26 = w[1];
    w[1] = vec4(_e26.x, 4.0, _e26.z, 3.0);
    uvec4 _e30 = eval_order();
    uvec4 _e32 = swizzle_views(1);
    uvec4 _e34 = a;
    uvec4 r = ((_e30 + _e32) + _e34);
    vec4 _e42 = w[1];
    _group_0_binding_0_cs[0] = (vec3(r.xyz) + _e42.xyz);
    return;
}

