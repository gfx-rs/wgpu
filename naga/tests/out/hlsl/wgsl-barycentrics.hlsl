struct FragmentInput_fs_main {
    float3 bary_3 : SV_Barycentrics;
};

struct FragmentInput_fs_main_no_perspective {
    noperspective float3 bary_4 : SV_Barycentrics;
};

struct FragmentInput_fs_main_both {
    float3 bary_5 : SV_Barycentrics;
    noperspective float3 bary_no_persp_1 : SV_Barycentrics1;
};

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    float3 bary = fragmentinput_fs_main.bary_3;
    return float4(bary, 1.0);
}

float4 fs_main_no_perspective(FragmentInput_fs_main_no_perspective fragmentinput_fs_main_no_perspective) : SV_Target0
{
    float3 bary_1 = fragmentinput_fs_main_no_perspective.bary_4;
    return float4(bary_1, 1.0);
}

float4 fs_main_both(FragmentInput_fs_main_both fragmentinput_fs_main_both) : SV_Target0
{
    float3 bary_2 = fragmentinput_fs_main_both.bary_5;
    float3 bary_no_persp = fragmentinput_fs_main_both.bary_no_persp_1;
    return float4(bary_2.xy, bary_no_persp.z, 1.0);
}
