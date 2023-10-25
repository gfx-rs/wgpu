struct type_1 {
    int member;
};

RWByteAddressBuffer unnamed : register(u0);

void function()
{
    int _expr3 = asint(unnamed.Load(0));
    unnamed.Store(0, asuint((_expr3 + 1)));
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    function();
}
