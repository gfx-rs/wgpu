struct type_2 {
    float member;
};

struct type_4 {
    uint2 member;
};

RWByteAddressBuffer global : register(u0);
ByteAddressBuffer global_1 : register(t1);
Texture2D<float> global_2 : register(t2);

void function()
{
    uint2 _e6 = asuint(global_1.Load2(0));
    float _e7 = global_2.Load(int3(_e6, 0)).x;
    global.Store(0, asuint((_e7).xxxx.x));
    return;
}

[numthreads(32, 1, 1)]
void cullfetch_depth()
{
    function();
}
