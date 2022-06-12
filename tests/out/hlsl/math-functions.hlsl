
void main()
{
    float4 v = (0.0).xxxx;
    float a = degrees(1.0);
    float b = radians(1.0);
    float4 c = degrees(v);
    float4 d = radians(v);
    int const_dot = dot(int2(0, 0), int2(0, 0));
    uint first_leading_bit_abs = firstbithigh(abs(0u));
}
