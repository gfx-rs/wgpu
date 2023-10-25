struct Ah {
    float inner[2];
};

ByteAddressBuffer ah : register(t0);

typedef float ret_Constructarray2_float_[2];
ret_Constructarray2_float_ Constructarray2_float_(float arg0, float arg1) {
    float ret[2] = { arg0, arg1 };
    return ret;
}

Ah ConstructAh(float arg0[2]) {
    Ah ret = (Ah)0;
    ret.inner = arg0;
    return ret;
}

[numthreads(1, 1, 1)]
void cs_main()
{
    Ah ah_1 = ConstructAh(Constructarray2_float_(asfloat(ah.Load(0+0)), asfloat(ah.Load(0+4))));
}
