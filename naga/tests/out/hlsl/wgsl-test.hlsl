RWByteAddressBuffer data : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    uint i = 0u;

    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e12 = i;
            i = (_e12 + 1u);
        }
        loop_init = false;
        uint _e2 = i;
        if ((_e2 < 4u)) {
        } else {
            break;
        }
        {
            uint _e6 = i;
            uint _e8 = i;
            data.Store(_e6*4, asuint((_e8 * 2u)));
        }
    }
    return;
}
