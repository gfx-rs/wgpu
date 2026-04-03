RWByteAddressBuffer out_ : register(u0);

int simple_for()
{
    int sum = int(0);

    for(int i = int(0); (i < int(10)); i = asint(asuint(i) + asuint(int(1)))) {
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

    while((i_1 < int(10))) {
        int _e6 = i_1;
        i_1 = asint(asuint(_e6) + asuint(int(1)));
    }
    int _e8 = i_1;
    return _e8;
}

int for_with_continue()
{
    int sum_1 = int(0);

    for(int i_2 = int(0); (i_2 < int(10)); i_2 = asint(asuint(i_2) + asuint(int(1)))) {
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

int nested_for_while()
{
    int sum_2 = int(0);
    int i_3 = int(0);

    while((i_3 < int(3))) {
        for(int j = int(0); (j < int(3)); j = asint(asuint(j) + asuint(int(1)))) {
            int _e12 = sum_2;
            int _e13 = i_3;
            int _e16 = j;
            sum_2 = asint(asuint(_e12) + asuint(asint(asuint(asint(asuint(_e13) * asuint(int(3)))) + asuint(_e16))));
        }
        int _e23 = i_3;
        i_3 = asint(asuint(_e23) + asuint(int(1)));
    }
    int _e25 = sum_2;
    return _e25;
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
    const int _e11 = nested_for_while();
    out_.Store(12, asuint(_e11));
    return;
}
