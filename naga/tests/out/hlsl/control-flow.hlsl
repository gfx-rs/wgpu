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
        }
        default: {
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
            }
            default: {
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
            break;
        }
    }
    int _expr4 = pos;
    switch(_expr4) {
        case 1: {
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
            break;
        }
        case 3:
        case 4: {
            pos = 2;
            break;
        }
        case 5: {
            pos = 3;
            break;
        }
        default:
        case 6: {
            pos = 4;
            break;
        }
    }
    switch(0u) {
        case 0u: {
            break;
        }
        default: {
            break;
        }
    }
    int _expr11 = pos;
    switch(_expr11) {
        case 1: {
            pos = 0;
            break;
        }
        case 2: {
            pos = 1;
            return;
        }
        case 3: {
            pos = 2;
            return;
        }
        case 4: {
            return;
        }
        default: {
            pos = 3;
            return;
        }
    }
}
