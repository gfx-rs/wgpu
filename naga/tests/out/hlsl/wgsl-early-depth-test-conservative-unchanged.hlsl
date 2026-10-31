struct FragmentInput_main {
    float4 pos_1 : SV_Position;
};

void main(FragmentInput_main fragmentinput_main)
{
    float4 pos = fragmentinput_main.pos_1;
    return;
}
