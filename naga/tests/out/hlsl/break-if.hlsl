void breakIfEmpty()
{
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            if (true) {
                break;
            }
        }
        loop_init = false;
    }
    return;
}

void breakIfEmptyBody(bool a)
{
    bool b = (bool)0;
    bool c = (bool)0;

    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            b = a;
            bool _e2 = b;
            c = (a != _e2);
            bool _e5 = c;
            if ((a == _e5)) {
                break;
            }
        }
        loop_init_1 = false;
    }
    return;
}

void breakIf(bool a_1)
{
    bool d = (bool)0;
    bool e = (bool)0;

    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            bool _e5 = e;
            if ((a_1 == _e5)) {
                break;
            }
        }
        loop_init_2 = false;
        d = a_1;
        bool _e2 = d;
        e = (a_1 != _e2);
    }
    return;
}

void breakIfSeparateVariable()
{
    uint counter = 0u;

    bool loop_init_3 = true;
    while(true) {
        if (!loop_init_3) {
            uint _e5 = counter;
            if ((_e5 == 5u)) {
                break;
            }
        }
        loop_init_3 = false;
        uint _e3 = counter;
        counter = (_e3 + 1u);
    }
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    return;
}
