RWByteAddressBuffer data : register(u0);

void main_1()
{
    uint phi_19_ = (uint)0;

    phi_19_ = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            phi_19_ = (phi_19_ + 1u);
        }
        loop_init = false;
        uint _e7 = phi_19_;
        bool should_continue = false;
        do {
            if ((_e7 < 4u)) {
            } else {
                break;
            }
            uint _e11 = asuint(data.Load(_e7*4+0));
            if ((_e11 == 1u)) {
                break;
            }
            uint _e13 = asuint(data.Load(_e7*4+0));
            if ((_e13 == 2u)) {
                break;
            }
            data.Store(_e7*4+0, asuint((_e7 * 2u)));
            break;
        } while(false);
        continue;
    }
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    main_1();
}
