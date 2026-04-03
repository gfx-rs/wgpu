RWByteAddressBuffer out_ : register(u0);

int simple_for()
{
    int sum = int(0);

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    for(int i = int(0); (i < int(10)); i = asint(asuint(i) + asuint(int(1)))) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        int _e7 = sum;
        int _e8 = i;
        sum = asint(asuint(_e7) + asuint(_e8));
    }
    int _e13 = sum;
    return _e13;
}

int simple_while()
{
    int i_1 = int(0);

    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    while((i_1 < int(10))) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        int _e6 = i_1;
        i_1 = asint(asuint(_e6) + asuint(int(1)));
    }
    int _e8 = i_1;
    return _e8;
}

int for_with_continue()
{
    int sum_1 = int(0);

    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
    for(int i_2 = int(0); (i_2 < int(10)); i_2 = asint(asuint(i_2) + asuint(int(1)))) {
        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        int _e7 = i_2;
        if ((_e7 == int(5))) {
            continue;
        }
        int _e10 = sum_1;
        int _e11 = i_2;
        sum_1 = asint(asuint(_e10) + asuint(_e11));
    }
    int _e16 = sum_1;
    return _e16;
}

int for_with_break()
{
    int sum_2 = int(0);

    uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
    for(int i_3 = int(0); (i_3 < int(10)); i_3 = asint(asuint(i_3) + asuint(int(1)))) {
        if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        int _e7 = i_3;
        if ((_e7 == int(5))) {
            break;
        }
        int _e10 = sum_2;
        int _e11 = i_3;
        sum_2 = asint(asuint(_e10) + asuint(_e11));
    }
    int _e16 = sum_2;
    return _e16;
}

int for_infinite()
{
    int i_4 = int(0);

    uint2 loop_bound_4 = uint2(4294967295u, 4294967295u);
    for(; ; ) {
        if (all(loop_bound_4 == uint2(0u, 0u))) { break; }
        loop_bound_4 -= uint2(loop_bound_4.y == 0u, 1u);
        int _e2 = i_4;
        if ((_e2 >= int(10))) {
            break;
        }
        int _e6 = i_4;
        i_4 = asint(asuint(_e6) + asuint(int(1)));
    }
    int _e8 = i_4;
    return _e8;
}

int for_no_update()
{
    int i_5 = int(0);

    uint2 loop_bound_5 = uint2(4294967295u, 4294967295u);
    for(; (i_5 < int(10)); ) {
        if (all(loop_bound_5 == uint2(0u, 0u))) { break; }
        loop_bound_5 -= uint2(loop_bound_5.y == 0u, 1u);
        int _e6 = i_5;
        i_5 = asint(asuint(_e6) + asuint(int(1)));
    }
    int _e8 = i_5;
    return _e8;
}

int while_with_break()
{
    int i_6 = int(0);

    uint2 loop_bound_6 = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound_6 == uint2(0u, 0u))) { break; }
        loop_bound_6 -= uint2(loop_bound_6.y == 0u, 1u);
        int _e3 = i_6;
        if ((_e3 >= int(10))) {
            break;
        }
        int _e7 = i_6;
        i_6 = asint(asuint(_e7) + asuint(int(1)));
    }
    int _e9 = i_6;
    return _e9;
}

int nested_loops()
{
    int sum_3 = int(0);
    int i_7 = int(0);

    uint2 loop_bound_7 = uint2(4294967295u, 4294967295u);
    while((i_7 < int(3))) {
        if (all(loop_bound_7 == uint2(0u, 0u))) { break; }
        loop_bound_7 -= uint2(loop_bound_7.y == 0u, 1u);
        uint2 loop_bound_8 = uint2(4294967295u, 4294967295u);
        for(int j = int(0); (j < int(3)); j = asint(asuint(j) + asuint(int(1)))) {
            if (all(loop_bound_8 == uint2(0u, 0u))) { break; }
            loop_bound_8 -= uint2(loop_bound_8.y == 0u, 1u);
            int _e12 = sum_3;
            int _e13 = i_7;
            int _e16 = j;
            sum_3 = asint(asuint(_e12) + asuint(asint(asuint(asint(asuint(_e13) * asuint(int(3)))) + asuint(_e16))));
        }
        int _e23 = i_7;
        i_7 = asint(asuint(_e23) + asuint(int(1)));
    }
    int _e25 = sum_3;
    return _e25;
}

int for_var_outside()
{
    int i_8 = int(0);

    uint2 loop_bound_9 = uint2(4294967295u, 4294967295u);
    for(; (i_8 < int(10)); i_8 = asint(asuint(i_8) + asuint(int(1)))) {
        if (all(loop_bound_9 == uint2(0u, 0u))) { break; }
        loop_bound_9 -= uint2(loop_bound_9.y == 0u, 1u);
    }
    int _e8 = i_8;
    return _e8;
}

[numthreads(1, 1, 1)]
void main()
{
    const int _e2 = simple_for();
    out_.Store(0, asuint(_e2));
    const int _e5 = simple_while();
    out_.Store(4, asuint(_e5));
    const int _e8 = for_with_continue();
    out_.Store(8, asuint(_e8));
    const int _e11 = for_with_break();
    out_.Store(12, asuint(_e11));
    const int _e14 = for_infinite();
    out_.Store(16, asuint(_e14));
    const int _e17 = for_no_update();
    out_.Store(20, asuint(_e17));
    const int _e20 = while_with_break();
    out_.Store(24, asuint(_e20));
    const int _e23 = nested_loops();
    out_.Store(28, asuint(_e23));
    const int _e26 = for_var_outside();
    out_.Store(32, asuint(_e26));
    return;
}
