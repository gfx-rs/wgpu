struct FragmentInput_main {
    float4 pos_1 : SV_Position;
};

void main(FragmentInput_main fragmentinput_main)
{
    float4 pos = float4(fragmentinput_main.pos_1.xyz, 1.0 / fragmentinput_main.pos_1.w);
    return;
}
