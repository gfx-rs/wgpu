#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 4, local_size_y = 2, local_size_z = 1) in;

struct Particle {
    vec3 position;
    uint hits;
    float history[4];
};
shared uint total;

shared uint small[3];

shared vec4 exact[8];

shared float grid[5][4];

shared Particle particles[2];


void main() {
    uint index = gl_LocalInvocationIndex;
    uint zero_init_index = 0u;
    if ((index < 3u)) {
        small[index] = 0u;
    }
    exact[index] = vec4(0.0);
    zero_init_index = index;
    while(true) {
        uint _e63 = zero_init_index;
        if ((_e63 >= 20u)) {
            break;
        }
        grid[(_e63 / 4u)][(_e63 - 4u * (_e63 / 4u))] = 0.0;
        zero_init_index = (_e63 + 8u);
    }
    if ((index < 2u)) {
        particles[index].position = vec3(0.0);
        particles[index].hits = 0u;
    }
    particles[(index / 4u)].history[(index - 4u * (index / 4u))] = 0.0;
    if ((index == 0u)) {
        total = 0u;
    }
    memoryBarrierShared();
    barrier();
    uint _e5 = small[(index - 3u * (index / 3u))];
    float _e10 = exact[index].x;
    float _e19 = grid[(index - 5u * (index / 5u))][(index - 4u * (index / 4u))];
    float _e27 = particles[(index - 2u * (index / 2u))].position.x;
    float _e37 = particles[(index - 2u * (index / 2u))].history[(index - 4u * (index / 4u))];
    float sum = ((((float(_e5) + _e10) + _e19) + _e27) + _e37);
    uint _e43 = atomicAdd(particles[0].hits, uint(sum));
    if ((index == 0u)) {
        uint _e50 = atomicOr(particles[1].hits, 0u);
        total = _e50;
        return;
    } else {
        return;
    }
}

