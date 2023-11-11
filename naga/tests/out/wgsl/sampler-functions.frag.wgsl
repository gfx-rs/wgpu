fn CalcShadowPCF1_(T_P_t_TextureDepth: texture_depth_2d, S_P_t_TextureDepth: sampler_comparison, t_ProjCoord: vec3<f32>) -> f32 {
    var t_ProjCoord_1: vec3<f32>;
    var t_Res: f32 = 0.0;

    t_ProjCoord_1 = t_ProjCoord;
    let _e6 = t_Res;
    let _e7 = t_ProjCoord_1;
    let _e9 = t_ProjCoord_1;
    let _e10 = _e9.xyz;
    let _e13 = textureSampleCompare(T_P_t_TextureDepth, S_P_t_TextureDepth, _e10.xy, _e10.z);
    t_Res = (_e6 + (_e13 * 0.2));
    let _e19 = t_Res;
    return _e19;
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
    let _e13 = t_ProjCoord_3;
    let _e15 = CalcShadowPCF1_(T_P_t_TextureDepth_1, S_P_t_TextureDepth_1, _e13.xyz);
    return _e15;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
