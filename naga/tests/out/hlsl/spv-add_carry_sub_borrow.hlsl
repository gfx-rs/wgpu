struct Output {
    uint sum;
    uint carry;
    uint diff;
    uint borrow;
};

struct Input {
    uint a;
    uint b;
};

RWByteAddressBuffer outp : register(u1);
RWByteAddressBuffer inp : register(u0);

Input ConstructInput(uint arg0, uint arg1) {
    Input ret = (Input)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

void main_1()
{
    uint c = (uint)0;
    uint d = (uint)0;

    uint _e5 = asuint(inp.Load(0));
    uint _e7 = asuint(inp.Load(4));
    uint _e8 = (_e5 + _e7);
    Input _e11 = ConstructInput(_e8, uint((_e8 < _e5)));
    c = _e11.b;
    outp.Store(0, asuint(_e11.a));
    uint _e15 = c;
    outp.Store(4, asuint(_e15));
    uint _e18 = asuint(inp.Load(0));
    uint _e20 = asuint(inp.Load(4));
    Input _e24 = ConstructInput((_e18 - _e20), uint((_e18 < _e20)));
    d = _e24.b;
    outp.Store(8, asuint(_e24.a));
    uint _e28 = d;
    outp.Store(12, asuint(_e28));
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    main_1();
}
