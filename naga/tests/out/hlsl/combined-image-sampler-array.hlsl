static float4 color = (float4)0;
Texture2D<float4> tex[2] : register(t1);
SamplerState _tex_sampler[2] : register(s1, space1);

void main_1()
{
    float4 _expr8 = tex[0].Sample(_tex_sampler[0], float2(0.0, 0.0));
    color = _expr8;
    float4 _expr9 = color;
    float4 _expr13 = tex[1].Sample(_tex_sampler[1], float2(0.0, 0.0));
    color = (_expr9 * _expr13);
    return;
}

float4 main() : SV_Target0
{
    main_1();
    float4 _expr1 = color;
    return _expr1;
}
