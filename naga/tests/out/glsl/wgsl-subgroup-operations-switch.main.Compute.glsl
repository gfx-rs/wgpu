#version 430 core
#extension GL_ARB_compute_shader : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_quad : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;


uint lowered(uint sid_1) {
    uint v = 0u;
    switch((sid_1 - 3u * (sid_1 / 3u))) {
        case 0u: {
            uint _e5 = subgroupBroadcastFirst(sid_1);
            v = _e5;
            break;
        }
        case 1u: {
            uint _e6 = subgroupAdd(sid_1);
            v = _e6;
            break;
        }
        default: {
            uint _e7 = subgroupBroadcastFirst(sid_1);
            v = _e7;
            break;
        }
    }
    uint _e8 = v;
    return _e8;
}

uint kept_fallthrough(uint sid_2) {
    uint v_1 = 0u;
    do {
        uint _e5 = subgroupBroadcastFirst(sid_2);
        v_1 = _e5;
    } while(false);
    uint _e6 = v_1;
    return _e6;
}

uint kept_break(uint sid_3) {
    uint v_2 = 0u;
    while(true) {
        switch((sid_3 - 2u * (sid_3 / 2u))) {
            case 0u: {
                uint _e5 = subgroupBroadcastFirst(sid_3);
                v_2 = _e5;
                break;
            }
            default: {
                v_2 = 1u;
                break;
            }
        }
        break;
    }
    uint _e7 = v_2;
    return _e7;
}

uint kept_no_subgroup_ops(uint sid_4) {
    uint v_3 = 0u;
    switch((sid_4 - 2u * (sid_4 / 2u))) {
        case 0u: {
            v_3 = 10u;
            break;
        }
        default: {
            v_3 = 20u;
            break;
        }
    }
    uint _e7 = v_3;
    return _e7;
}

void main() {
    uint sid = gl_SubgroupInvocationID;
    uint _e1 = lowered(sid);
    uint _e2 = kept_fallthrough(sid);
    uint _e3 = kept_break(sid);
    uint _e4 = kept_no_subgroup_ops(sid);
    return;
}

