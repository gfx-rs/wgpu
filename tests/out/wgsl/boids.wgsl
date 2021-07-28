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
var<storage> particlesSrc: Particles;
[[group(0), binding(2)]]
var<storage,read_write> particlesDst: Particles;

[[stage(compute), workgroup_size(64, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    var vPos: vec2<f32>;
    var vVel: vec2<f32>;
    var cMass: vec2<f32>;
    var cVel: vec2<f32>;
    var colVel: vec2<f32>;
    var cMassCount: i32 = 0;
    var cVelCount: i32 = 0;
    var pos: vec2<f32>;
    var vel: vec2<f32>;
    var i: u32 = 0u;

    let index: u32 = global_invocation_id.x;
    if ((index >= NUM_PARTICLES)) {
        return;
    }
    let _e10: vec2<f32> = particlesSrc.particles[index].pos;
    vPos = _e10;
    let _e15: vec2<f32> = particlesSrc.particles[index].vel;
    vVel = _e15;
    cMass = vec2<f32>(0.0, 0.0);
    cVel = vec2<f32>(0.0, 0.0);
    colVel = vec2<f32>(0.0, 0.0);
    loop {
        let _e37: u32 = i;
        if ((_e37 >= NUM_PARTICLES)) {
            break;
        }
        let _e39: u32 = i;
        if ((_e39 == index)) {
            continue;
        }
        let _e42: u32 = i;
        let _e45: vec2<f32> = particlesSrc.particles[_e42].pos;
        pos = _e45;
        let _e47: u32 = i;
        let _e50: vec2<f32> = particlesSrc.particles[_e47].vel;
        vel = _e50;
        let _e51: vec2<f32> = pos;
        let _e52: vec2<f32> = vPos;
        let _e55: f32 = params.rule1Distance;
        if ((distance(_e51, _e52) < _e55)) {
            let _e57: vec2<f32> = cMass;
            let _e58: vec2<f32> = pos;
            cMass = (_e57 + _e58);
            let _e60: i32 = cMassCount;
            cMassCount = (_e60 + 1);
        }
        let _e63: vec2<f32> = pos;
        let _e64: vec2<f32> = vPos;
        let _e67: f32 = params.rule2Distance;
        if ((distance(_e63, _e64) < _e67)) {
            let _e69: vec2<f32> = colVel;
            let _e70: vec2<f32> = pos;
            let _e71: vec2<f32> = vPos;
            colVel = (_e69 - (_e70 - _e71));
        }
        let _e74: vec2<f32> = pos;
        let _e75: vec2<f32> = vPos;
        let _e78: f32 = params.rule3Distance;
        if ((distance(_e74, _e75) < _e78)) {
            let _e80: vec2<f32> = cVel;
            let _e81: vec2<f32> = vel;
            cVel = (_e80 + _e81);
            let _e83: i32 = cVelCount;
            cVelCount = (_e83 + 1);
        }
        continuing {
            let _e86: u32 = i;
            i = (_e86 + 1u);
        }
    }
    let _e89: i32 = cMassCount;
    if ((_e89 > 0)) {
        let _e92: vec2<f32> = cMass;
        let _e93: i32 = cMassCount;
        let _e97: vec2<f32> = vPos;
        cMass = ((_e92 / vec2<f32>(f32(_e93))) - _e97);
    }
    let _e99: i32 = cVelCount;
    if ((_e99 > 0)) {
        let _e102: vec2<f32> = cVel;
        let _e103: i32 = cVelCount;
        cVel = (_e102 / vec2<f32>(f32(_e103)));
    }
    let _e107: vec2<f32> = vVel;
    let _e108: vec2<f32> = cMass;
    let _e110: f32 = params.rule1Scale;
    let _e113: vec2<f32> = colVel;
    let _e115: f32 = params.rule2Scale;
    let _e118: vec2<f32> = cVel;
    let _e120: f32 = params.rule3Scale;
    vVel = (((_e107 + (_e108 * _e110)) + (_e113 * _e115)) + (_e118 * _e120));
    let _e123: vec2<f32> = vVel;
    let _e125: vec2<f32> = vVel;
    vVel = (normalize(_e123) * clamp(length(_e125), 0.0, 0.1));
    let _e131: vec2<f32> = vPos;
    let _e132: vec2<f32> = vVel;
    let _e134: f32 = params.deltaT;
    vPos = (_e131 + (_e132 * _e134));
    let _e137: vec2<f32> = vPos;
    if ((_e137.x < -1.0)) {
        vPos.x = 1.0;
    }
    let _e143: vec2<f32> = vPos;
    if ((_e143.x > 1.0)) {
        vPos.x = -1.0;
    }
    let _e149: vec2<f32> = vPos;
    if ((_e149.y < -1.0)) {
        vPos.y = 1.0;
    }
    let _e155: vec2<f32> = vPos;
    if ((_e155.y > 1.0)) {
        vPos.y = -1.0;
    }
    let _e164: vec2<f32> = vPos;
    particlesDst.particles[index].pos = _e164;
    let _e168: vec2<f32> = vVel;
    particlesDst.particles[index].vel = _e168;
    return;
}
