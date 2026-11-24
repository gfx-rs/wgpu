bool ZeroValuebool() {
    return (bool)0;
}

[numthreads(1, 1, 1)]
void main()
{
    bool a = (bool)0;
    bool b = (bool)0;
    bool c = (bool)0;
    bool a_1 = (bool)0;
    bool d = (bool)0;
    bool e = (bool)0;
    uint counter = (uint)0;
    uint2 loop_bound = (4294967295u).xx;
    uint2 loop_bound_1 = (4294967295u).xx;
    uint2 loop_bound_2 = (4294967295u).xx;
    uint2 loop_bound_3 = (4294967295u).xx;

    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            if (true) {
                break;
            }
        }
        loop_init = false;
        {
            uint2 _e41 = loop_bound;
            loop_bound = (_e41 - uint2(((_e41.y == 0u) ? 1u : 0u), 1u));
            if (all((_e41 == (0u).xx))) {
                break;
            }
        }
    }
    a = false;
    bool _e4 = a;
    b = ZeroValuebool();
    c = ZeroValuebool();
    bool loop_init_1 = true;
    while(true) {
        if (!loop_init_1) {
            b = _e4;
            bool _e6 = b;
            c = (_e4 != _e6);
            bool _e9 = c;
            if ((_e4 == _e9)) {
                break;
            }
        }
        loop_init_1 = false;
        {
            uint2 _e53 = loop_bound_1;
            loop_bound_1 = (_e53 - uint2(((_e53.y == 0u) ? 1u : 0u), 1u));
            if (all((_e53 == (0u).xx))) {
                break;
            }
        }
    }
    a_1 = false;
    bool _e16 = a_1;
    d = ZeroValuebool();
    e = ZeroValuebool();
    bool loop_init_2 = true;
    while(true) {
        if (!loop_init_2) {
            bool _e21 = e;
            if ((_e16 == _e21)) {
                break;
            }
        }
        loop_init_2 = false;
        {
            uint2 _e65 = loop_bound_2;
            loop_bound_2 = (_e65 - uint2(((_e65.y == 0u) ? 1u : 0u), 1u));
            if (all((_e65 == (0u).xx))) {
                break;
            }
        }
        d = _e16;
        bool _e18 = d;
        e = (_e16 != _e18);
    }
    counter = 0u;
    bool loop_init_3 = true;
    while(true) {
        if (!loop_init_3) {
            uint _e32 = counter;
            if ((_e32 == 5u)) {
                break;
            }
        }
        loop_init_3 = false;
        {
            uint2 _e77 = loop_bound_3;
            loop_bound_3 = (_e77 - uint2(((_e77.y == 0u) ? 1u : 0u), 1u));
            if (all((_e77 == (0u).xx))) {
                break;
            }
        }
        uint _e29 = counter;
        counter = (_e29 + 1u);
    }
    return;
}
