RWTexture2D<float> s_r_r : register(u0);
RWTexture2D<float4> s_rg_r : register(u1);
RWTexture2D<float4> s_rgba_r : register(u2);
RWTexture2D<float> s_r_w : register(u0, space1);
RWTexture2D<float4> s_rg_w : register(u1, space1);
RWTexture2D<float4> s_rgba_w : register(u2, space1);

float4 LoadedStorageValueFromfloat(float arg) {float4 ret = float4(arg, 0.0, 0.0, 1.0);return ret;}
[numthreads(1, 1, 1)]
void csLoad()
{
    float4 phony = LoadedStorageValueFromfloat(s_r_r.Load((0u).xx));
    float4 phony_1 = s_rg_r.Load((0u).xx);
    float4 phony_2 = s_rgba_r.Load((0u).xx);
    return;
}

[numthreads(1, 1, 1)]
void csStore()
{
    s_r_w[(0u).xx] = (0.0).xxxx;
    s_rg_w[(0u).xx] = (0.0).xxxx;
    s_rgba_w[(0u).xx] = (0.0).xxxx;
    return;
}
