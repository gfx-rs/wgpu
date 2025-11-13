struct FragmentInput_main {
    uint view_index_1 : SV_ViewID;
};

void main(FragmentInput_main fragmentinput_main)
{
    uint view_index = fragmentinput_main.view_index_1;
    return;
}
