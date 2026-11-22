void f_u0028_b1_u003b(inout bool cond)
{
    uint2 loop_bound = (uint2)0;

    loop_bound = (4294967295u).xx;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _e1 = cond;
            if (!(_e1)) {
                break;
            }
        }
        loop_init = false;
        uint2 _e9 = loop_bound;
        if (all((_e9 == (0u).xx))) {
            break;
        }
        loop_bound = (_e9 - uint2(((_e9.y == 0u) ? 1u : 0u), 1u));
        continue;
    }
    return;
}

void main_1()
{
    bool param = (bool)0;

    param = false;
    f_u0028_b1_u003b(param);
    return;
}

void main()
{
    main_1();
}
