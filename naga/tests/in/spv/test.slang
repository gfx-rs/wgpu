[vk::binding(0, 0)]
RWStructuredBuffer<uint32_t> data;

[shader("compute")]
void main()
{
    for (uint32_t i = 0; i < 4; i++)
    {
        if (data[i] == 1)
        {
            continue;
        }
        if (data[i] == 2)
        {
            break;
        }
        data[i] = i * 2; // Example operation: double each element
    }
}
