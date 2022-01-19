struct Particle {
    pos: vec2<f32>;
    vel: vec2<f32>;
};

struct SimParams {
    deltaT: f32;
    rule1Distance: f32;
    rule2Distance: f32;
    rule3Distance: f32;
    rule1Scale: f32;
    rule2Scale: f32;
    rule3Scale: f32;
};

struct Particles {
    particles: @stride(16) array<Particle>;
};

let NUM_PARTICLES: u32 = 1500u;

@group(0) @binding(0) 
var<uniform> params: SimParams;
@group(0) @binding(1) 
var<storage> particlesSrc: Particles;
@group(0) @binding(2) 
var<storage, read_write> particlesDst: Particles;

@stage(compute) @workgroup_size(64, 1, 1) 
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
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

    let index = global_invocation_id.x;
    if ((index >= NUM_PARTICLES)) {
        return;
    }
    let _e10 = particlesSrc.particles[index].pos;
    vPos = _e10;
    let _e15 = particlesSrc.particles[index].vel;
    vVel = _e15;
    cMass = vec2<f32>(0.0, 0.0);
    cVel = vec2<f32>(0.0, 0.0);
    colVel = vec2<f32>(0.0, 0.0);
    loop {
        let _e37 = i;
        if ((_e37 >= NUM_PARTICLES)) {
            break;
        }
        let _e39 = i;
        if ((_e39 == index)) {
            continue;
        }
        let _e42 = i;
        let _e45 = particlesSrc.particles[_e42].pos;
        pos = _e45;
        let _e47 = i;
        let _e50 = particlesSrc.particles[_e47].vel;
        vel = _e50;
        let _e51 = pos;
        let _e52 = vPos;
        let _e55 = params.rule1Distance;
        if ((distance(_e51, _e52) < _e55)) {
            let _e57 = cMass;
            let _e58 = pos;
            cMass = (_e57 + _e58);
            let _e60 = cMassCount;
            cMassCount = (_e60 + 1);
        }
        let _e63 = pos;
        let _e64 = vPos;
        let _e67 = params.rule2Distance;
        if ((distance(_e63, _e64) < _e67)) {
            let _e69 = colVel;
            let _e70 = pos;
            let _e71 = vPos;
            colVel = (_e69 - (_e70 - _e71));
        }
        let _e74 = pos;
        let _e75 = vPos;
        let _e78 = params.rule3Distance;
        if ((distance(_e74, _e75) < _e78)) {
            let _e80 = cVel;
            let _e81 = vel;
            cVel = (_e80 + _e81);
            let _e83 = cVelCount;
            cVelCount = (_e83 + 1);
        }
        continuing {
            let _e86 = i;
            i = (_e86 + 1u);
        }
    }
    let _e89 = cMassCount;
    if ((_e89 > 0)) {
        let _e92 = cMass;
        let _e93 = cMassCount;
        let _e97 = vPos;
        cMass = ((_e92 / vec2<f32>(f32(_e93))) - _e97);
    }
    let _e99 = cVelCount;
    if ((_e99 > 0)) {
        let _e102 = cVel;
        let _e103 = cVelCount;
        cVel = (_e102 / vec2<f32>(f32(_e103)));
    }
    let _e107 = vVel;
    let _e108 = cMass;
    let _e110 = params.rule1Scale;
    let _e113 = colVel;
    let _e115 = params.rule2Scale;
    let _e118 = cVel;
    let _e120 = params.rule3Scale;
    vVel = (((_e107 + (_e108 * _e110)) + (_e113 * _e115)) + (_e118 * _e120));
    let _e123 = vVel;
    let _e125 = vVel;
    vVel = (normalize(_e123) * clamp(length(_e125), 0.0, 0.10000000149011612));
    let _e131 = vPos;
    let _e132 = vVel;
    let _e134 = params.deltaT;
    vPos = (_e131 + (_e132 * _e134));
    let _e138 = vPos.x;
    if ((_e138 < -1.0)) {
        vPos.x = 1.0;
    }
    let _e144 = vPos.x;
    if ((_e144 > 1.0)) {
        vPos.x = -1.0;
    }
    let _e150 = vPos.y;
    if ((_e150 < -1.0)) {
        vPos.y = 1.0;
    }
    let _e156 = vPos.y;
    if ((_e156 > 1.0)) {
        vPos.y = -1.0;
    }
    let _e164 = vPos;
    particlesDst.particles[index].pos = _e164;
    let _e168 = vVel;
    particlesDst.particles[index].vel = _e168;
    return;
}
