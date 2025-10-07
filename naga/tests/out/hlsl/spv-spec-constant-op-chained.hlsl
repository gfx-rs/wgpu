struct type_2 {
    uint member;
    uint member_1;
    uint member_2;
};

static const int _spec_const_op_12_ = int(13);
static const int _spec_const_op_13_ = int(39);
static const int _spec_const_op_14_ = int(26);
static const int _spec_const_op_15_ = int(13);
static const int _spec_const_op_16_ = int(130);
static const int _spec_const_op_const_17_ = int(1);
static const int _spec_const_op_18_ = int(129);
static const int _spec_const_op_19_ = int(387);

RWByteAddressBuffer global : register(u0);

void function()
{
    global.Store(0, asuint(asuint(_spec_const_op_13_)));
    global.Store(4, asuint(asuint(_spec_const_op_14_)));
    global.Store(8, asuint(asuint(_spec_const_op_19_)));
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    function();
}
