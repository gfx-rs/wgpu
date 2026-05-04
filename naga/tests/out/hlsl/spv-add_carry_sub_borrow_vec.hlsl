struct Output {
    uint2 sum;
    uint2 carry;
    uint2 diff;
    uint2 borrow;
};

struct Input {
    uint2 a;
    uint2 b;
};

RWByteAddressBuffer outp : register(u1);
RWByteAddressBuffer inp : register(u0);

Input ConstructInput(uint2 arg0, uint2 arg1) {
    Input ret = (Input)0;
    ret.a = arg0;
    ret.b = arg1;
    return ret;
}

void main_1()
{
    uint2 c = (uint2)0;
    uint2 d = (uint2)0;

    uint2 _e5 = asuint(inp.Load2(0));
    uint2 _e7 = asuint(inp.Load2(8));
    uint2 _e8 = (_e5 + _e7);
    Input _e11 = ConstructInput(_e8, uint2((_e8 < _e5)));
    c = _e11.b;
    outp.Store2(0, asuint(_e11.a));
    uint2 _e15 = c;
    outp.Store2(8, asuint(_e15));
    uint2 _e18 = asuint(inp.Load2(0));
    uint2 _e20 = asuint(inp.Load2(8));
    Input _e24 = ConstructInput((_e18 - _e20), uint2((_e18 < _e20)));
    d = _e24.b;
    outp.Store2(16, asuint(_e24.a));
    uint2 _e28 = d;
    outp.Store2(24, asuint(_e28));
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    main_1();
}
