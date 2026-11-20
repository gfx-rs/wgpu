struct FragmentInput_fs_main {
    float3 bary_2 : SV_Barycentrics;
};

struct FragmentInput_fs_main_no_perspective {
    noperspective float3 bary_3 : SV_Barycentrics;
};

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    float3 bary = fragmentinput_fs_main.bary_2;
    return float4(bary.x, bary.y, bary.z, 1.0);
}

float4 fs_main_no_perspective(FragmentInput_fs_main_no_perspective fragmentinput_fs_main_no_perspective) : SV_Target0
{
    float3 bary_1 = fragmentinput_fs_main_no_perspective.bary_3;
    return float4(bary_1.x, bary_1.y, bary_1.z, 1.0);
}
