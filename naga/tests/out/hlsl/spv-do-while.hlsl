bool ZeroValuebool() {
    return (bool)0;
}

void main()
{
    bool param = (bool)0;
    uint2 loop_bound = (4294967295u).xx;

    param = ZeroValuebool();
    param = false;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _e2 = param;
            if (!(_e2)) {
                break;
            }
        }
        loop_init = false;
        {
            uint2 _e11 = loop_bound;
            loop_bound = (_e11 - uint2(((_e11.y == 0u) ? 1u : 0u), 1u));
            if (all((_e11 == (0u).xx))) {
                break;
            }
        }
        continue;
    }
}
