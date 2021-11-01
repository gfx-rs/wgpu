
struct type1 {
    int member;
};

RWByteAddressBuffer global : register(u0);

void function()
{
    int _expr8 = asint(global.Load(0));
    global.Store(0, asuint((_expr8 + 1)));
    return;
}

[numthreads(64, 1, 1)]
void main()
{
    function();
}
