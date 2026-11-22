void breakIfEmpty()
{
    uint2 loop_bound = (uint2)0;

    loop_bound = (4294967295u).xx;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            if (true) {
                break;
            }
        }
        loop_init = false;
        uint2 _e7 = loop_bound;
        if (all((_e7 == (0u).xx))) {
            break;
        }
        loop_bound = (_e7 - uint2(((_e7.y == 0u) ? 1u : 0u), 1u));
    }
    return;
}

void breakIfEmptyBody(bool a)
{
    bool b = (bool)0;
    bool c = (bool)0;
    uint2 loop_bound_1 = (uint2)0;

    loop_bound_1 = (4294967295u).xx;
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
        uint2 _e13 = loop_bound_1;
        if (all((_e13 == (0u).xx))) {
            break;
        }
        loop_bound_1 = (_e13 - uint2(((_e13.y == 0u) ? 1u : 0u), 1u));
    }
    return;
}

void breakIf(bool a_1)
{
    bool d = (bool)0;
    bool e = (bool)0;
    uint2 loop_bound_2 = (uint2)0;

    loop_bound_2 = (4294967295u).xx;
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            bool _e5 = e;
            if ((a_1 == _e5)) {
                break;
            }
        }
        loop_init_2 = false;
        uint2 _e13 = loop_bound_2;
        if (all((_e13 == (0u).xx))) {
            break;
        }
        loop_bound_2 = (_e13 - uint2(((_e13.y == 0u) ? 1u : 0u), 1u));
        d = a_1;
        bool _e2 = d;
        e = (a_1 != _e2);
    }
    return;
}

void breakIfSeparateVariable()
{
    uint counter = 0u;
    uint2 loop_bound_3 = (uint2)0;

    loop_bound_3 = (4294967295u).xx;
    bool loop_init_3 = true;
    while(true) {
        if (!loop_init_3) {
            uint _e5 = counter;
            if ((_e5 == 5u)) {
                break;
            }
        }
        loop_init_3 = false;
        uint2 _e14 = loop_bound_3;
        if (all((_e14 == (0u).xx))) {
            break;
        }
        loop_bound_3 = (_e14 - uint2(((_e14.y == 0u) ? 1u : 0u), 1u));
        uint _e2 = counter;
        counter = (_e2 + 1u);
    }
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    breakIfEmpty();
    breakIfEmptyBody(false);
    breakIf(false);
    breakIfSeparateVariable();
    return;
}
