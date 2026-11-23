cbuffer binding : register(b0) { float binding; }

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    int local = (int)0;
    int local_1 = (int)0;

    float phony = binding;
    float phony_1 = binding;
    local = int(5);
    int phony_2 = local;
    local_1 = int(5);
    int b = local_1;
    float phony_3 = binding;
    return;
}
