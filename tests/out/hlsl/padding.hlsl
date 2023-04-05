struct S {
    float3 a;
    int _end_pad_0;
};

struct Test {
    S a;
    float b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

struct Test2_ {
    float3 a[2];
    int _pad1_0;
    float b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

struct Test3_ {
    row_major float4x3 a;
    int _pad1_0;
    float b;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

cbuffer input1_ : register(b0) { Test input1_; }
cbuffer input2_ : register(b1) { Test2_ input2_; }
cbuffer input3_ : register(b2) { Test3_ input3_; }

float4 vertex() : SV_Position
{
    float _expr4 = input1_.b;
    float _expr8 = input2_.b;
    float _expr12 = input3_.b;
    return ((((1.0).xxxx * _expr4) * _expr8) * _expr12);
}
