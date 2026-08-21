#version 460
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_basic : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

uint _4(uint _6, uint _7)
{
    return _6 % ((_7 == 0u) ? 1u : _7);
}

uint _17(uint _16)
{
    uint _20 = 0u;
    switch (_4(_16, 3u))
    {
        case 0u:
        {
            _20 = subgroupBroadcastFirst(_16);
            break;
        }
        case 1u:
        {
            _20 = subgroupAdd(_16);
            break;
        }
        default:
        {
            _20 = subgroupBroadcastFirst(_16);
            break;
        }
    }
    return _20;
}

uint _34(uint _33)
{
    uint _36 = 0u;
    switch (_4(_33, 2u))
    {
        default:
        {
            _36 = subgroupBroadcastFirst(_33);
            break;
        }
    }
    return _36;
}

uint _45(uint _44)
{
    uint _46 = 0u;
    uvec2 _58 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(uvec2(0u), _58)))
        {
            break;
        }
        _58 -= uvec2(uint(_58.y == 0u), 1u);
        switch (_4(_44, 2u))
        {
            case 0u:
            {
                _46 = subgroupBroadcastFirst(_44);
                break;
            }
            default:
            {
                _46 = 1u;
                break;
            }
        }
        break;
    }
    return _46;
}

uint _77(uint _76)
{
    uint _80 = 0u;
    switch (_4(_76, 2u))
    {
        case 0u:
        {
            _80 = 10u;
            break;
        }
        default:
        {
            _80 = 20u;
            break;
        }
    }
    return _80;
}

void main()
{
}

