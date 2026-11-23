bool ZeroValuebool() {
    return (bool)0;
}

void main()
{
    bool param = (bool)0;
    uint2 loop_bound = (uint2)0;

    param = ZeroValuebool();
    param = false;
    loop_bound = (4294967295u).xx;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            bool _e2 = param;
            if (!(_e2)) {
                break;
            }
        }
        loop_init = false;
        uint2 _e12 = loop_bound;
        if (all((_e12 == (0u).xx))) {
            break;
        }
        loop_bound = (_e12 - uint2(((_e12.y == 0u) ? 1u : 0u), 1u));
        continue;
    }
}
