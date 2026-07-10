ByteAddressBuffer input : register(t0);
RWByteAddressBuffer output : register(u1);

float4 ZeroValuefloat4() {
    return (float4)0;
}

[numthreads(1, 1, 1)]
void main()
{
    uint t = 0u;
    float4 acc_noinit = (float4)0;
    float4 acc_init = (float4)0;
    uint d = (uint)0;

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e47 = t;
            t = (_e47 + 1u);
        }
        loop_init = false;
        uint _e2 = t;
        if ((_e2 < 4u)) {
        } else {
            break;
        }
        {
            acc_noinit = ZeroValuefloat4();
            acc_init = ZeroValuefloat4();
            d = 0u;
            uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
            bool loop_init_1 = true;
            while(true) {
                if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
                loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
                if (!loop_init_1) {
                    uint _e28 = d;
                    d = (_e28 + 1u);
                }
                loop_init_1 = false;
                uint _e11 = d;
                if ((_e11 < 16u)) {
                } else {
                    break;
                }
                {
                    uint _e15 = t;
                    uint _e18 = d;
                    float _e21 = asfloat(input.Load(((_e15 * 16u) + _e18)*4));
                    float4 v = (_e21).xxxx;
                    float4 _e23 = acc_noinit;
                    acc_noinit = (_e23 + v);
                    float4 _e25 = acc_init;
                    acc_init = (_e25 + v);
                }
            }
            uint _e31 = t;
            float _e36 = acc_noinit.x;
            output.Store((_e31 * 2u)*4, asuint(_e36));
            uint _e38 = t;
            float _e45 = acc_init.x;
            output.Store(((_e38 * 2u) + 1u)*4, asuint(_e45));
        }
    }
    return;
}
