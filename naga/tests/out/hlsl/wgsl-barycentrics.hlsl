struct FragmentInput_fs_main {
    float3 bary_1 : SV_Barycentrics;
};

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    float3 bary = fragmentinput_fs_main.bary_1;
    return float4(bary, 1.0);
}
