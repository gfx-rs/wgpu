
void switch_default_break(int i)
{
    switch(i) {
        default: {
            break;
        }
    }
}

void switch_case_break()
{
    switch(0) {
        case 0: {
            break;
            break;
        }
    }
    return;
}

void loop_switch_continue(int x)
{
    while(true) {
        switch(x) {
            case 1: {
                continue;
                break;
            }
        }
    }
    return;
}

[numthreads(1, 1, 1)]
void main(uint3 global_id : SV_DispatchThreadID)
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
