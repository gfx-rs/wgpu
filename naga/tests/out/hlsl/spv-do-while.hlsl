bool ZeroValuebool() {
    return (bool)0;
}

void main()
{
    bool param = (bool)0;

    param = ZeroValuebool();
    param = false;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            bool _e2 = param;
            if (!(_e2)) {
                break;
            }
        }
        loop_init = false;
        continue;
    }
}
