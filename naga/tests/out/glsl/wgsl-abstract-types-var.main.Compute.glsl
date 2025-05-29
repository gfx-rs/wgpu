#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

ivec2 xvipaiai_1 = ivec2(42, 43);

uvec2 xvupaiai_1 = uvec2(44u, 45u);

vec2 xvfpaiai_1 = vec2(46.0, 47.0);

vec2 xvfpafaf_1 = vec2(48.0, 49.0);

vec2 xvfpaiaf_1 = vec2(48.0, 49.0);

uvec2 xvupuai_2 = uvec2(42u, 43u);

uvec2 xvupaiu_2 = uvec2(42u, 43u);

uvec2 xvuuai_2 = uvec2(42u, 43u);

uvec2 xvuaiu_2 = uvec2(42u, 43u);

ivec2 xvip_1 = ivec2(0, 0);

uvec2 xvup_1 = uvec2(0u, 0u);

vec2 xvfp_1 = vec2(0.0, 0.0);

mat2x2 xmfp_1 = mat2x2(vec2(0.0, 0.0), vec2(0.0, 0.0));

mat2x2 xmfpaiaiaiai_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));

mat2x2 xmfpafaiaiai_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));

mat2x2 xmfpaiafaiai_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));

mat2x2 xmfpaiaiafai_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));

mat2x2 xmfpaiaiaiaf_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));

ivec2 xvispai_1 = ivec2(1);

vec2 xvfspaf_1 = vec2(1.0);

ivec2 xvis_ai_1 = ivec2(1);

uvec2 xvus_ai_1 = uvec2(1u);

vec2 xvfs_ai_1 = vec2(1.0);

vec2 xvfs_af_1 = vec2(1.0);

float xafafaf_1[2] = float[2](1.0, 2.0);

float xafaiai_1[2] = float[2](1.0, 2.0);

int xaipaiai_1[2] = int[2](1, 2);

uint xaupaiai[2] = uint[2](1u, 2u);

float xafpaiaf_1[2] = float[2](1.0, 2.0);

float xafpafai_1[2] = float[2](1.0, 2.0);

float xafpafaf_1[2] = float[2](1.0, 2.0);

ivec3 xavipai_1[1] = ivec3[1](ivec3(1));

vec3 xavfpai_1[1] = vec3[1](vec3(1.0));

vec3 xavfpaf_1[1] = vec3[1](vec3(1.0));

ivec2 xvisai_1 = ivec2(1);

uvec2 xvusai_1 = uvec2(1u);

vec2 xvfsai_1 = vec2(1.0);

vec2 xvfsaf_1 = vec2(1.0);

ivec2 ivispai = ivec2(1);

vec2 ivfspaf = vec2(1.0);

ivec2 ivis_ai = ivec2(1);

uvec2 ivus_ai = uvec2(1u);

vec2 ivfs_ai = vec2(1.0);

vec2 ivfs_af = vec2(1.0);

float iafafaf[2] = float[2](1.0, 2.0);

float iafaiai[2] = float[2](1.0, 2.0);

int iaipaiai_1[2] = int[2](1, 2);

float iafpafaf_1[2] = float[2](1.0, 2.0);

float iafpaiaf_1[2] = float[2](1.0, 2.0);

float iafpafai_1[2] = float[2](1.0, 2.0);

ivec3 iavipai[1] = ivec3[1](ivec3(1));

ivec3 iavfpai[1] = ivec3[1](ivec3(1));

vec3 iavfpaf[1] = vec3[1](vec3(1.0));


void globals() {
    ivec2 phony = xvipaiai_1;
    uvec2 phony_1 = xvupaiai_1;
    vec2 phony_2 = xvfpaiai_1;
    vec2 phony_3 = xvfpafaf_1;
    vec2 phony_4 = xvfpaiaf_1;
    uvec2 phony_5 = xvupuai_2;
    uvec2 phony_6 = xvupaiu_2;
    uvec2 phony_7 = xvuuai_2;
    uvec2 phony_8 = xvuaiu_2;
    ivec2 phony_9 = xvip_1;
    uvec2 phony_10 = xvup_1;
    vec2 phony_11 = xvfp_1;
    mat2x2 phony_12 = xmfp_1;
    mat2x2 phony_13 = xmfpaiaiaiai_1;
    mat2x2 phony_14 = xmfpafaiaiai_1;
    mat2x2 phony_15 = xmfpaiafaiai_1;
    mat2x2 phony_16 = xmfpaiaiafai_1;
    mat2x2 phony_17 = xmfpaiaiaiaf_1;
    ivec2 phony_18 = xvispai_1;
    vec2 phony_19 = xvfspaf_1;
    ivec2 phony_20 = xvis_ai_1;
    uvec2 phony_21 = xvus_ai_1;
    vec2 phony_22 = xvfs_ai_1;
    vec2 phony_23 = xvfs_af_1;
    float phony_24[2] = xafafaf_1;
    float phony_25[2] = xafaiai_1;
    int phony_26[2] = xaipaiai_1;
    uint phony_27[2] = xaupaiai;
    float phony_28[2] = xafpaiaf_1;
    float phony_29[2] = xafpafai_1;
    float phony_30[2] = xafpafaf_1;
    ivec3 phony_31[1] = xavipai_1;
    vec3 phony_32[1] = xavfpai_1;
    vec3 phony_33[1] = xavfpaf_1;
    ivec2 phony_34 = xvisai_1;
    uvec2 phony_35 = xvusai_1;
    vec2 phony_36 = xvfsai_1;
    vec2 phony_37 = xvfsaf_1;
    ivec2 phony_38 = ivispai;
    vec2 phony_39 = ivfspaf;
    ivec2 phony_40 = ivis_ai;
    uvec2 phony_41 = ivus_ai;
    vec2 phony_42 = ivfs_ai;
    vec2 phony_43 = ivfs_af;
    float phony_44[2] = iafafaf;
    float phony_45[2] = iafaiai;
    int phony_46[2] = iaipaiai_1;
    float phony_47[2] = iafpafaf_1;
    float phony_48[2] = iafpaiaf_1;
    float phony_49[2] = iafpafai_1;
    ivec3 phony_50[1] = iavipai;
    ivec3 phony_51[1] = iavfpai;
    vec3 phony_52[1] = iavfpaf;
    return;
}

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
    xvipaiai = ivec2(42, 43);
    xvupaiai = uvec2(44u, 45u);
    xvfpaiai = vec2(46.0, 47.0);
    xvfpafaf = vec2(48.0, 49.0);
    xvfpaiaf = vec2(48.0, 49.0);
    xvupuai = uvec2(42u, 43u);
    xvupaiu = uvec2(42u, 43u);
    xvuuai = uvec2(42u, 43u);
    xvuaiu = uvec2(42u, 43u);
    xvip = ivec2(0, 0);
    xvup = uvec2(0u, 0u);
    xvfp = vec2(0.0, 0.0);
    xmfp = mat2x2(vec2(0.0, 0.0), vec2(0.0, 0.0));
    xmfpaiaiaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpafaiaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpaiafaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpaiaiafai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpaiaiaiaf = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfp_faiaiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpai_faiai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpaiai_fai = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xmfpaiaiai_f = mat2x2(vec2(1.0, 2.0), vec2(3.0, 4.0));
    xvispai = ivec2(1);
    xvfspaf = vec2(1.0);
    xvis_ai = ivec2(1);
    xvus_ai = uvec2(1u);
    xvfs_ai = vec2(1.0);
    xvfs_af = vec2(1.0);
    xafafaf = float[2](1.0, 2.0);
    xaf_faf = float[2](1.0, 2.0);
    xafaf_f = float[2](1.0, 2.0);
    xafaiai = float[2](1.0, 2.0);
    xai_iai = int[2](1, 2);
    xaiai_i = int[2](1, 2);
    xaipaiai = int[2](1, 2);
    xafpaiai = float[2](1.0, 2.0);
    xafpaiaf = float[2](1.0, 2.0);
    xafpafai = float[2](1.0, 2.0);
    xafpafaf = float[2](1.0, 2.0);
    xavipai = ivec3[1](ivec3(1));
    xavfpai = vec3[1](vec3(1.0));
    xavfpaf = vec3[1](vec3(1.0));
    xvisai = ivec2(1);
    xvusai = uvec2(1u);
    xvfsai = vec2(1.0);
    xvfsaf = vec2(1.0);
    iaipaiai = int[2](1, 2);
    iafpaiaf = float[2](1.0, 2.0);
    iafpafai = float[2](1.0, 2.0);
    iafpafaf = float[2](1.0, 2.0);
    return;
}

void mixed_constant_and_runtime_arguments() {
    uint u = 0u;
    int i = 0;
    float f = 0.0;
    uvec2 xvupuai_1 = uvec2(0u);
    uvec2 xvupaiu_1 = uvec2(0u);
    vec2 xvfpfai = vec2(0.0);
    vec2 xvfpfaf = vec2(0.0);
    uvec2 xvuuai_1 = uvec2(0u);
    uvec2 xvuaiu_1 = uvec2(0u);
    mat2x2 xmfp_faiaiai_1 = mat2x2(0.0);
    mat2x2 xmfpai_faiai_1 = mat2x2(0.0);
    mat2x2 xmfpaiai_fai_1 = mat2x2(0.0);
    mat2x2 xmfpaiaiai_f_1 = mat2x2(0.0);
    float xaf_faf_1[2] = float[2](0.0, 0.0);
    float xafaf_f_1[2] = float[2](0.0, 0.0);
    float xaf_fai[2] = float[2](0.0, 0.0);
    float xafai_f[2] = float[2](0.0, 0.0);
    int xai_iai_1[2] = int[2](0, 0);
    int xaiai_i_1[2] = int[2](0, 0);
    float xafp_faf[2] = float[2](0.0, 0.0);
    float xafpaf_f[2] = float[2](0.0, 0.0);
    float xafp_fai[2] = float[2](0.0, 0.0);
    float xafpai_f[2] = float[2](0.0, 0.0);
    int xaip_iai[2] = int[2](0, 0);
    int xaipai_i[2] = int[2](0, 0);
    ivec2 xvisi = ivec2(0);
    uvec2 xvusu = uvec2(0u);
    vec2 xvfsf = vec2(0.0);
    uint _e3 = u;
    xvupuai_1 = uvec2(_e3, 43u);
    uint _e7 = u;
    xvupaiu_1 = uvec2(42u, _e7);
    float _e11 = f;
    xvfpfai = vec2(_e11, 47.0);
    float _e15 = f;
    xvfpfaf = vec2(_e15, 49.0);
    uint _e19 = u;
    xvuuai_1 = uvec2(_e19, 43u);
    uint _e23 = u;
    xvuaiu_1 = uvec2(42u, _e23);
    float _e27 = f;
    xmfp_faiaiai_1 = mat2x2(vec2(_e27, 2.0), vec2(3.0, 4.0));
    float _e35 = f;
    xmfpai_faiai_1 = mat2x2(vec2(1.0, _e35), vec2(3.0, 4.0));
    float _e43 = f;
    xmfpaiai_fai_1 = mat2x2(vec2(1.0, 2.0), vec2(_e43, 4.0));
    float _e51 = f;
    xmfpaiaiai_f_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, _e51));
    float _e59 = f;
    xaf_faf_1 = float[2](_e59, 2.0);
    float _e63 = f;
    xafaf_f_1 = float[2](1.0, _e63);
    float _e67 = f;
    xaf_fai = float[2](_e67, 2.0);
    float _e71 = f;
    xafai_f = float[2](1.0, _e71);
    int _e75 = i;
    xai_iai_1 = int[2](_e75, 2);
    int _e79 = i;
    xaiai_i_1 = int[2](1, _e79);
    float _e83 = f;
    xafp_faf = float[2](_e83, 2.0);
    float _e87 = f;
    xafpaf_f = float[2](1.0, _e87);
    float _e91 = f;
    xafp_fai = float[2](_e91, 2.0);
    float _e95 = f;
    xafpai_f = float[2](1.0, _e95);
    int _e99 = i;
    xaip_iai = int[2](_e99, 2);
    int _e103 = i;
    xaipai_i = int[2](1, _e103);
    int _e107 = i;
    xvisi = ivec2(_e107);
    uint _e110 = u;
    xvusu = uvec2(_e110);
    float _e113 = f;
    xvfsf = vec2(_e113);
    uint _e116 = u;
    xvupuai_1 = uvec2(_e116, 43u);
    uint _e119 = u;
    xvupaiu_1 = uvec2(42u, _e119);
    uint _e122 = u;
    xvuuai_1 = uvec2(_e122, 43u);
    uint _e125 = u;
    xvuaiu_1 = uvec2(42u, _e125);
    float _e128 = f;
    xmfp_faiaiai_1 = mat2x2(vec2(_e128, 2.0), vec2(3.0, 4.0));
    float _e135 = f;
    xmfpai_faiai_1 = mat2x2(vec2(1.0, _e135), vec2(3.0, 4.0));
    float _e142 = f;
    xmfpaiai_fai_1 = mat2x2(vec2(1.0, 2.0), vec2(_e142, 4.0));
    float _e149 = f;
    xmfpaiaiai_f_1 = mat2x2(vec2(1.0, 2.0), vec2(3.0, _e149));
    float _e156 = f;
    xaf_faf_1 = float[2](_e156, 2.0);
    float _e159 = f;
    xafaf_f_1 = float[2](1.0, _e159);
    float _e162 = f;
    xaf_fai = float[2](_e162, 2.0);
    float _e165 = f;
    xafai_f = float[2](1.0, _e165);
    int _e168 = i;
    xai_iai_1 = int[2](_e168, 2);
    int _e171 = i;
    xaiai_i_1 = int[2](1, _e171);
    float _e174 = f;
    xafp_faf = float[2](_e174, 2.0);
    float _e177 = f;
    xafpaf_f = float[2](1.0, _e177);
    float _e180 = f;
    xafp_fai = float[2](_e180, 2.0);
    float _e183 = f;
    xafpai_f = float[2](1.0, _e183);
    int _e186 = i;
    xaip_iai = int[2](_e186, 2);
    int _e189 = i;
    xaipai_i = int[2](1, _e189);
    int _e192 = i;
    xvisi = ivec2(_e192);
    uint _e194 = u;
    xvusu = uvec2(_e194);
    float _e196 = f;
    xvfsf = vec2(_e196);
    return;
}

void main() {
    globals();
    all_constant_arguments();
    mixed_constant_and_runtime_arguments();
    return;
}

