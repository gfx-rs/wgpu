#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void all_constant_arguments() {
    ivec2 xvipaiai = ivec2(42, 43);
    uvec2 xvupaiai = uvec2(44u, 45u);
    vec2 xvfpaiai = vec2(46.0, 47.0);
    vec2 xvfpafaf = vec2(48.0, 49.0);
    vec2 xvfpaiaf = vec2(48.0, 49.0);
    uvec2 xvupuai = uvec2(42u, 43u);
    uvec2 xvupaiu = uvec2(42u, 43u);
    uvec2 xvuuai = uvec2(42u, 43u);
    uvec2 xvuaiu = uvec2(42u, 43u);
    ivec2 xvip = ivec2(0, 0);
    uvec2 xvup = uvec2(0u, 0u);
    vec2 xvfp = vec2(0.0, 0.0);
    mat2x2 xmfp = mat2x2(vec2(0.0, 0.0), vec2(0.0, 0.0));
    mat2x2 xmfpaiaiaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpafaiaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpaiafaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpaiaiafai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpaiaiaiaf = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfp_faiaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpai_faiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpaiai_fai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    mat2x2 xmfpaiaiai_f = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    ivec2 xvispai = ivec2(1);
    vec2 xvfspaf = vec2(1.0);
    ivec2 xvis_ai = ivec2(1);
    uvec2 xvus_ai = uvec2(1u);
    vec2 xvfs_ai = vec2(1.0);
    vec2 xvfs_af = vec2(1.0);
    float xafafaf[2] = float[2](1.0, 2.0);
    float xaf_faf[2] = float[2](1.0, 2.0);
    float xafaf_f[2] = float[2](1.0, 2.0);
    float xafaiai[2] = float[2](1.0, 2.0);
    int xai_iai[2] = int[2](1, 2);
    int xaiai_i[2] = int[2](1, 2);
    int xaipaiai[2] = int[2](1, 2);
    float xafpaiai[2] = float[2](1.0, 2.0);
    float xafpaiaf[2] = float[2](1.0, 2.0);
    float xafpafai[2] = float[2](1.0, 2.0);
    float xafpafaf[2] = float[2](1.0, 2.0);
    ivec3 xavipai[1] = ivec3[1](ivec3(1));
    vec3 xavfpai[1] = vec3[1](vec3(1.0));
    vec3 xavfpaf[1] = vec3[1](vec3(1.0));
    ivec2 xvisai = ivec2(1);
    uvec2 xvusai = uvec2(1u);
    vec2 xvfsai = vec2(1.0);
    vec2 xvfsaf = vec2(1.0);
    int iaipaiai[2] = int[2](1, 2);
    float iafpaiaf[2] = float[2](1.0, 2.0);
    float iafpafai[2] = float[2](1.0, 2.0);
    float iafpafaf[2] = float[2](1.0, 2.0);
    return;
}

void mixed_constant_and_runtime_arguments() {
    uint u = 0u;
    int i = 0;
    float f = 0.0;
    uint _e3 = u;
    uvec2 xvupuai_1 = uvec2(_e3, 43u);
    uint _e6 = u;
    uvec2 xvupaiu_1 = uvec2(42u, _e6);
    float _e9 = f;
    vec2 xvfpfai = vec2(_e9, 47.0);
    float _e12 = f;
    vec2 xvfpfaf = vec2(_e12, 49.0);
    uint _e15 = u;
    uvec2 xvuuai_1 = uvec2(_e15, 43u);
    uint _e18 = u;
    uvec2 xvuaiu_1 = uvec2(42u, _e18);
    float _e21 = f;
    mat2x2 xmfp_faiaiai_1 = mat2x2(vec2(_e21, 2.0), vec2(3.0, 4.0));
    float _e28 = f;
    mat2x2 xmfpai_faiai_1 = mat2x2(vec2(1.0, _e28), vec2(3.0, 4.0));
    float _e35 = f;
    mat2x2 xmfpaiai_fai_1 = mat2x2(vec2(1.0, 2.0), vec2(_e35, 4.0));
    float _e42 = f;
    mat2x2 xmfpaiaiai_f_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, _e42));
    float _e49 = f;
    float xaf_faf_1[2] = float[2](_e49, 2.0);
    float _e52 = f;
    float xafaf_f_1[2] = float[2](1.0, _e52);
    float _e55 = f;
    float xaf_fai[2] = float[2](_e55, 2.0);
    float _e58 = f;
    float xafai_f[2] = float[2](1.0, _e58);
    int _e61 = i;
    int xai_iai_1[2] = int[2](_e61, 2);
    int _e64 = i;
    int xaiai_i_1[2] = int[2](1, _e64);
    float _e67 = f;
    float xafp_faf[2] = float[2](_e67, 2.0);
    float _e70 = f;
    float xafpaf_f[2] = float[2](1.0, _e70);
    float _e73 = f;
    float xafp_fai[2] = float[2](_e73, 2.0);
    float _e76 = f;
    float xafpai_f[2] = float[2](1.0, _e76);
    int _e79 = i;
    int xaip_iai[2] = int[2](_e79, 2);
    int _e82 = i;
    int xaipai_i[2] = int[2](1, _e82);
    int _e85 = i;
    ivec2 xvisi = ivec2(_e85);
    uint _e87 = u;
    uvec2 xvusu = uvec2(_e87);
    float _e89 = f;
    vec2 xvfsf = vec2(_e89);
    return;
}

void main() {
    all_constant_arguments();
    mixed_constant_and_runtime_arguments();
    return;
}

