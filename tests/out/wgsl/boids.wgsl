struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
}

struct SimParams {
    deltaT: f32,
    rule1Distance: f32,
    rule2Distance: f32,
    rule3Distance: f32,
    rule1Scale: f32,
    rule2Scale: f32,
    rule3Scale: f32,
}

struct Particles {
    particles: array<Particle>,
}

const NUM_PARTICLES: u32 = 1500u;

@group(0) @binding(0) 
var<uniform> params: SimParams;
@group(0) @binding(1) 
var<storage> particlesSrc: Particles;
@group(0) @binding(2) 
var<storage, read_write> particlesDst: Particles;

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    var vPos: vec2<f32>;
    var vVel: vec2<f32>;
    var cMass: vec2<f32> = vec2<f32>(0.0, 0.0);
    var cVel: vec2<f32> = vec2<f32>(0.0, 0.0);
    var colVel: vec2<f32> = vec2<f32>(0.0, 0.0);
    var cMassCount: i32 = 0;
    var cVelCount: i32 = 0;
    var pos: vec2<f32>;
    var vel: vec2<f32>;
    var i: u32 = 0u;

    let index = global_invocation_id.x;
    if (index >= NUM_PARTICLES) {
        return;
    }
    let _e8 = particlesSrc.particles[index].pos;
    vPos = _e8;
    let _e14 = particlesSrc.particles[index].vel;
    vVel = _e14;
    loop {
        let _e36 = i;
        if (_e36 >= NUM_PARTICLES) {
            break;
        }
        let _e39 = i;
        if (_e39 == index) {
            continue;
        }
        let _e43 = i;
        let _e46 = particlesSrc.particles[_e43].pos;
        pos = _e46;
        let _e49 = i;
        let _e52 = particlesSrc.particles[_e49].vel;
        vel = _e52;
        let _e53 = pos;
        let _e54 = vPos;
        let _e58 = params.rule1Distance;
        if (distance(_e53, _e54) < _e58) {
            let _e60 = cMass;
            let _e61 = pos;
            cMass = (_e60 + _e61);
            let _e63 = cMassCount;
            cMassCount = (_e63 + 1);
        }
        let _e66 = pos;
        let _e67 = vPos;
        let _e71 = params.rule2Distance;
        if (distance(_e66, _e67) < _e71) {
            let _e73 = colVel;
            let _e74 = pos;
            let _e75 = vPos;
            colVel = (_e73 - (_e74 - _e75));
        }
        let _e78 = pos;
        let _e79 = vPos;
        let _e83 = params.rule3Distance;
        if (distance(_e78, _e79) < _e83) {
            let _e85 = cVel;
            let _e86 = vel;
            cVel = (_e85 + _e86);
            let _e88 = cVelCount;
            cVelCount = (_e88 + 1);
        }
        continuing {
            let _e91 = i;
            i = (_e91 + 1u);
        }
    }
    let _e94 = cMassCount;
    if (_e94 > 0) {
        let _e97 = cMass;
        let _e98 = cMassCount;
        let _e102 = vPos;
        cMass = ((_e97 / vec2(f32(_e98))) - _e102);
    }
    let _e104 = cVelCount;
    if (_e104 > 0) {
        let _e107 = cVel;
        let _e108 = cVelCount;
        cVel = (_e107 / vec2(f32(_e108)));
    }
    let _e112 = vVel;
    let _e113 = cMass;
    let _e116 = params.rule1Scale;
    let _e119 = colVel;
    let _e122 = params.rule2Scale;
    let _e125 = cVel;
    let _e128 = params.rule3Scale;
    vVel = (((_e112 + (_e113 * _e116)) + (_e119 * _e122)) + (_e125 * _e128));
    let _e131 = vVel;
    let _e133 = vVel;
    vVel = (normalize(_e131) * clamp(length(_e133), 0.0, 0.1));
    let _e139 = vPos;
    let _e140 = vVel;
    let _e143 = params.deltaT;
    vPos = (_e139 + (_e140 * _e143));
    let _e147 = vPos.x;
    if (_e147 < -1.0) {
        vPos.x = 1.0;
    }
    let _e153 = vPos.x;
    if (_e153 > 1.0) {
        vPos.x = -1.0;
    }
    let _e159 = vPos.y;
    if (_e159 < -1.0) {
        vPos.y = 1.0;
    }
    let _e165 = vPos.y;
    if (_e165 > 1.0) {
        vPos.y = -1.0;
    }
    let _e174 = vPos;
    particlesDst.particles[index].pos = _e174;
    let _e179 = vVel;
    particlesDst.particles[index].vel = _e179;
    return;
}
