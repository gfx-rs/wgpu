struct type_4 {
    float4 member : SV_Position;
    float member_1;
    float member_2[1];
    float member_3[1];
    int _end_pad_0;
};

type_4 Constructtype_4(float4 arg0, float arg1, float arg2[1], float arg3[1]) {
    type_4 ret = (type_4)0;
    ret.member = arg0;
    ret.member_1 = arg1;
    ret.member_2 = arg2;
    ret.member_3 = arg3;
    return ret;
}

typedef float ret_ZeroValuearray1_float_[1];
ret_ZeroValuearray1_float_ ZeroValuearray1_float_() {
    return (float[1])0;
}

static type_4 global = Constructtype_4(float4(0.0, 0.0, 0.0, 1.0), 1.0, ZeroValuearray1_float_(), ZeroValuearray1_float_());
static int global_1 = (int)0;

void function()
{
    int _e9 = global_1;
    global.member = float4(((_e9 == 0) ? -4.0 : 1.0), ((_e9 == 2) ? 4.0 : -1.0), 0.0, 1.0);
    return;
}

float4 main(uint param : SV_VertexID) : SV_Position
{
    global_1 = int(param);
    function();
    float _e6 = global.member.y;
    global.member.y = -(_e6);
    float4 _e8 = global.member;
    return _e8;
}
