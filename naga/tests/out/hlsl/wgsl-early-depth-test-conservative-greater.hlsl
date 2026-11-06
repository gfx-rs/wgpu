struct FragmentInput_main {
    noperspective centroid float4 pos_1 : SV_Position;
};

float main(FragmentInput_main fragmentinput_main) : SV_DepthGreaterEqual
{
    float4 pos = float4(fragmentinput_main.pos_1.xyz, 1.0 / fragmentinput_main.pos_1.w);
    return (pos.z + 0.1);
}
