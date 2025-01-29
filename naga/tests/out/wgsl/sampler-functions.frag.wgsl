fn CalcShadowPCF1_(T_P_t_TextureDepth: texture_depth_2d, S_P_t_TextureDepth: sampler_comparison, t_ProjCoord: vec3<f32>) -> f32 {
    var t_ProjCoord_1: vec3<f32>;
    var t_Res: f32 = 0f;

    t_ProjCoord_1 = t_ProjCoord;
    let _e6 = t_Res;
    let _e7 = t_ProjCoord_1;
    let _e8 = _e7.xyz;
    let _e11 = textureSampleCompare(T_P_t_TextureDepth, S_P_t_TextureDepth, _e8.xy, _e8.z);
    t_Res = (_e6 + (_e11 * 0.2f));
    let _e17 = t_Res;
    return _e17;
}

fn CalcShadowPCF(T_P_t_TextureDepth_1: texture_depth_2d, S_P_t_TextureDepth_1: sampler_comparison, t_ProjCoord_2: vec3<f32>, t_Bias: f32) -> f32 {
    var t_ProjCoord_3: vec3<f32>;
    var t_Bias_1: f32;

    t_ProjCoord_3 = t_ProjCoord_2;
    t_Bias_1 = t_Bias;
    let _e7 = t_ProjCoord_3;
    let _e9 = t_Bias_1;
    t_ProjCoord_3.z = (_e7.z + _e9);
    let _e11 = t_ProjCoord_3;
    let _e13 = CalcShadowPCF1_(T_P_t_TextureDepth_1, S_P_t_TextureDepth_1, _e11.xyz);
    return _e13;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
