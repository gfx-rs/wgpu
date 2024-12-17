cbuffer binding : register(b0) { float binding; }

int five()
{
    return 5;
}

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float phony = binding;
    float phony_1 = binding;
    const int _e6 = five();
    const int _e7 = five();
    float phony_2 = binding;
}
