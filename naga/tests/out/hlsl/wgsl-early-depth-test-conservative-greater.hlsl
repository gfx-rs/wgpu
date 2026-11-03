struct FragmentInput_main {
    noperspective centroid float4 pos_1 : SV_Position;
};

float main(FragmentInput_main fragmentinput_main) : SV_DepthGreaterEqual
{
    float4 pos = fragmentinput_main.pos_1;
    return (pos.z + 0.1);
}
