struct S {
    bool b;
};

struct V {
    bool2 b;
};

struct Middle {
    S s;
    float x;
    V v;
    float y;
};

struct Last {
    float x;
    S s;
};

struct ArrayMiddle {
    S a[3];
    int _pad1_0;
    int _pad1_1;
    float x;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

static Middle middle = (Middle)0;
static Last last = (Last)0;
static ArrayMiddle array_middle = (ArrayMiddle)0;
RWByteAddressBuffer o : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    float x = 0.0;
    bool local = (bool)0;
    bool local_1 = (bool)0;
    bool local_2 = (bool)0;

    float _e4 = asfloat(o.Load(0));
    middle.s.b = (_e4 > 1.0);
    float _e11 = asfloat(o.Load(0));
    last.s.b = (_e11 > 2.0);
    float _e19 = asfloat(o.Load(0));
    array_middle.a[1].b = (_e19 > 3.0);
    bool _e27 = middle.s.b;
    if (_e27) {
        bool _e34 = middle.v.b.y;
        local = _e34;
    } else {
        local = false;
    }
    bool _e36 = local;
    if (_e36) {
        bool _e42 = last.s.b;
        local_1 = _e42;
    } else {
        local_1 = false;
    }
    bool _e44 = local_1;
    if (_e44) {
        bool _e51 = array_middle.a[1].b;
        local_2 = _e51;
    } else {
        local_2 = false;
    }
    bool _e53 = local_2;
    if (_e53) {
        float _e56 = middle.x;
        float _e59 = middle.y;
        float _e63 = last.x;
        float _e67 = array_middle.x;
        x = (((_e56 + _e59) + _e63) + _e67);
    }
    float _e70 = x;
    o.Store(0, asuint(_e70));
    return;
}
