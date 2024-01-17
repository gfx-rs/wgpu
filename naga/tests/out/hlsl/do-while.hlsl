void fb1_(inout bool cond)
{
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _expr1 = cond;
            if (!(_expr1)) {
                break;
            }
        }
        loop_init = false;
        continue;
    }
    return;
}

void main_1()
{
    bool param = (bool)0;

    param = false;
    fb1_(param);
    return;
}

void main()
{
    main_1();
}
