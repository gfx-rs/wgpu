struct FragmentInput_fs_main {
    nointerpolation float v_1 : LOC0;
};

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    float v[3] = { GetAttributeAtVertex(fragmentinput_fs_main.v_1, 0), GetAttributeAtVertex(fragmentinput_fs_main.v_1, 1), GetAttributeAtVertex(fragmentinput_fs_main.v_1, 2) };
    return float4(v[0], v[1], v[2], 1.0);
}
