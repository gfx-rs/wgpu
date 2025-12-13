struct FragmentIn {
    float value : LOC1;
    float value2_ : LOC3;
    float4 position : SV_Position;
};

struct FragmentInput_fs_main {
    float value : LOC1;
    float value2_ : LOC3;
    float4 position : SV_Position;
};

float4 fs_main(FragmentInput_fs_main fragmentinput_fs_main) : SV_Target0
{
    FragmentIn v_out = { fragmentinput_fs_main.value, fragmentinput_fs_main.value2_, fragmentinput_fs_main.position };
    return float4(v_out.value, v_out.value, v_out.value2_, v_out.value2_);
}
