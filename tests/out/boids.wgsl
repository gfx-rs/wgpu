struct Particle {
    pos: vec2<f32>;
    vel: vec2<f32>;
};

[[block]]
struct SimParams {
    deltaT: f32;
    rule1Distance: f32;
    rule2Distance: f32;
    rule3Distance: f32;
    rule1Scale: f32;
    rule2Scale: f32;
    rule3Scale: f32;
};

[[block]]
struct Particles {
    particles: [[stride(16)]] array<Particle>;
};

let NUM_PARTICLES: u32 = 1500u;

[[group(0), binding(0)]]
var<uniform> params: SimParams;
[[group(0), binding(1)]]
var<storage> particlesSrc: [[access(read)]] Particles;
[[group(0), binding(2)]]
var<storage> particlesDst: [[access(read_write)]] Particles;

[[stage(compute), workgroup_size(64, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    var vPos: vec2<f32>;
    var vVel: vec2<f32>;
    var cMass: vec2<f32>;
    var cVel: vec2<f32>;
    var colVel: vec2<f32>;
    var cMassCount: i32 = 0;
    var cVelCount: i32 = 0;
    var pos1: vec2<f32>;
    var vel1: vec2<f32>;
    var i: u32 = 0u;

    let index: u32 = global_invocation_id.x;
    if ((index >= NUM_PARTICLES)) {
        return;
    }
    vPos = particlesSrc.particles[index].pos;
    vVel = particlesSrc.particles[index].vel;
    cMass = vec2<f32>(0.0, 0.0);
    cVel = vec2<f32>(0.0, 0.0);
    colVel = vec2<f32>(0.0, 0.0);
    loop {
        if ((i >= NUM_PARTICLES)) {
            break;
        }
        if ((i == index)) {
            continue;
        }
        pos1 = particlesSrc.particles[i].pos;
        vel1 = particlesSrc.particles[i].vel;
        if ((distance(pos1, vPos) < params.rule1Distance)) {
            cMass = (cMass + pos1);
            cMassCount = (cMassCount + 1);
        }
        if ((distance(pos1, vPos) < params.rule2Distance)) {
            colVel = (colVel - (pos1 - vPos));
        }
        if ((distance(pos1, vPos) < params.rule3Distance)) {
            cVel = (cVel + vel1);
            cVelCount = (cVelCount + 1);
        }
        continuing {
            i = (i + 1u);
        }
    }
    if ((cMassCount > 0)) {
        cMass = ((cMass / vec2<f32>(f32(cMassCount))) - vPos);
    }
    if ((cVelCount > 0)) {
        cVel = (cVel / vec2<f32>(f32(cVelCount)));
    }
    vVel = (((vVel + (cMass * params.rule1Scale)) + (colVel * params.rule2Scale)) + (cVel * params.rule3Scale));
    vVel = (normalize(vVel) * clamp(length(vVel), 0.0, 0.1));
    vPos = (vPos + (vVel * params.deltaT));
    if ((vPos.x < -1.0)) {
        vPos.x = 1.0;
    }
    if ((vPos.x > 1.0)) {
        vPos.x = -1.0;
    }
    if ((vPos.y < -1.0)) {
        vPos.y = 1.0;
    }
    if ((vPos.y > 1.0)) {
        vPos.y = -1.0;
    }
    particlesDst.particles[index].pos = vPos;
    particlesDst.particles[index].vel = vVel;
    return;
}
