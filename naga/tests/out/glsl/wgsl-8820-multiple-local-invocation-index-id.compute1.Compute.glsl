#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct Input {
    uvec3 local_invocation_id;
    uint local_invocation_index;
};
shared uint wg_var;


void main() {
    if (gl_LocalInvocationID == uvec3(0u)) {
        wg_var = 0u;
    }
    memoryBarrierShared();
    barrier();
    Input input_ = Input(gl_LocalInvocationID, gl_LocalInvocationIndex);
    wg_var = (input_.local_invocation_index * 2u);
    uint _e6 = wg_var;
    wg_var = (_e6 + input_.local_invocation_id.x);
    return;
}

