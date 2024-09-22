cbuffer binding : register(b0) { float binding; }

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    float _phony_2 = binding;
    int _phony_3 = 5;
}
