
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
            bool _expr2 = b;
            c = (a != _expr2);
            bool _expr5 = c;
            bool unnamed = (a == _expr5);
            if (unnamed) {
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
            bool _expr5 = e;
            bool unnamed_1 = (a_1 == _expr5);
            if (unnamed_1) {
                break;
            }
        }
        loop_init_2 = false;
        d = a_1;
        bool _expr2 = d;
        e = (a_1 != _expr2);
    }
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    return;
}
