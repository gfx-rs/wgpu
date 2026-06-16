#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

uint _5()
{
    int _18 = int(1u);
    int _19 = int(2u);
    uint _39 = (((0u + (bitfieldExtract(3u, int(0u), int(8u)) * bitfieldExtract(4u, int(0u), int(8u)))) + (bitfieldExtract(3u, int(8u), int(8u)) * bitfieldExtract(4u, int(8u), int(8u)))) + (bitfieldExtract(3u, int(16u), int(8u)) * bitfieldExtract(4u, int(16u), int(8u)))) + (bitfieldExtract(3u, int(24u), int(8u)) * bitfieldExtract(4u, int(24u), int(8u)));
    int _59 = int(5u + _39);
    int _60 = int(6u + _39);
    uint _76 = 7u + _39;
    uint _77 = 8u + _39;
    return (((0u + (bitfieldExtract(_76, int(0u), int(8u)) * bitfieldExtract(_77, int(0u), int(8u)))) + (bitfieldExtract(_76, int(8u), int(8u)) * bitfieldExtract(_77, int(8u), int(8u)))) + (bitfieldExtract(_76, int(16u), int(8u)) * bitfieldExtract(_77, int(16u), int(8u)))) + (bitfieldExtract(_76, int(24u), int(8u)) * bitfieldExtract(_77, int(24u), int(8u)));
}

void main()
{
}

