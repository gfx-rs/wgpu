uint test_packed_integer_dot_product()
{
    int c_5_ = dot4add_i8packed(1u, 2u, 0);
    uint c_6_ = dot4add_u8packed(3u, 4u, 0);
    uint _e7 = (5u + c_6_);
    uint _e9 = (6u + c_6_);
    int c_7_ = dot4add_i8packed(_e7, _e9, 0);
    uint _e12 = (7u + c_6_);
    uint _e14 = (8u + c_6_);
    uint c_8_ = dot4add_u8packed(_e12, _e14, 0);
    return c_8_;
}

[numthreads(1, 1, 1)]
void main()
{
    const uint _e0 = test_packed_integer_dot_product();
    return;
}
