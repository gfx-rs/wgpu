uint test_packed_integer_dot_product()
{
    int c_5_ = dot(int4(1u, 1u >> 8, 1u >> 16, 1u >> 24) << 24 >> 24, int4(2u, 2u >> 8, 2u >> 16, 2u >> 24) << 24 >> 24);
    uint c_6_ = dot(uint4(3u, 3u >> 8, 3u >> 16, 3u >> 24) << 24 >> 24, uint4(4u, 4u >> 8, 4u >> 16, 4u >> 24) << 24 >> 24);
    uint _e7 = (5u + c_6_);
    uint _e9 = (6u + c_6_);
    int c_7_ = dot(int4(_e7, _e7 >> 8, _e7 >> 16, _e7 >> 24) << 24 >> 24, int4(_e9, _e9 >> 8, _e9 >> 16, _e9 >> 24) << 24 >> 24);
    uint _e12 = (7u + c_6_);
    uint _e14 = (8u + c_6_);
    uint c_8_ = dot(uint4(_e12, _e12 >> 8, _e12 >> 16, _e12 >> 24) << 24 >> 24, uint4(_e14, _e14 >> 8, _e14 >> 16, _e14 >> 24) << 24 >> 24);
    return c_8_;
}

[numthreads(1, 1, 1)]
void main()
{
    const uint _e0 = test_packed_integer_dot_product();
    return;
}
