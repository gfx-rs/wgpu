static float4 color = (float4)0;
Texture2D<float4> tex : register(t1);
SamplerState _tex_sampler : register(s1, space1);

void main_1()
{
    float4 _expr4 = tex.Sample(_tex_sampler, float2(0.0, 0.0));
    color = _expr4;
    return;
}

float4 main() : SV_Target0
{
    main_1();
    float4 _expr1 = color;
    return _expr1;
}
