
struct ComputeInput_main {
    uint3 global_id1 : SV_DispatchThreadID;
};

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    int pos = (int)0;

    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    switch(1) {
        default: {
            pos = 1;
        }
    }
    int _expr4 = pos;
    switch(_expr4) {
        case 1: {
            pos = 0;
            break;
            break;
        }
        case 2: {
            pos = 1;
            return;
            break;
        }
        case 3: {
            /* fallthrough */
            {
                pos = 2;
            }
        }
        case 4: {
            return;
            break;
        }
        default: {
            pos = 3;
            return;
        }
    }
}
