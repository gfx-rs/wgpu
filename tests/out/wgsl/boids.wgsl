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
var<storage, read_write> particlesDst: Particles;

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
    let e10: vec2<f32> = particlesSrc.particles[index].pos;
    vPos = e10;
    let e15: vec2<f32> = particlesSrc.particles[index].vel;
    vVel = e15;
    cMass = vec2<f32>(0.0, 0.0);
    cVel = vec2<f32>(0.0, 0.0);
    colVel = vec2<f32>(0.0, 0.0);
    loop {
        let e37: u32 = i;
        if ((e37 >= NUM_PARTICLES)) {
            break;
        }
        let e39: u32 = i;
        if ((e39 == index)) {
            continue;
        }
        let e42: u32 = i;
        let e45: vec2<f32> = particlesSrc.particles[e42].pos;
        pos = e45;
        let e47: u32 = i;
        let e50: vec2<f32> = particlesSrc.particles[e47].vel;
        vel = e50;
        let e51: vec2<f32> = pos;
        let e52: vec2<f32> = vPos;
        let e55: f32 = params.rule1Distance;
        if ((distance(e51, e52) < e55)) {
            let e57: vec2<f32> = cMass;
            let e58: vec2<f32> = pos;
            cMass = (e57 + e58);
            let e60: i32 = cMassCount;
            cMassCount = (e60 + 1);
        }
        let e63: vec2<f32> = pos;
        let e64: vec2<f32> = vPos;
        let e67: f32 = params.rule2Distance;
        if ((distance(e63, e64) < e67)) {
            let e69: vec2<f32> = colVel;
            let e70: vec2<f32> = pos;
            let e71: vec2<f32> = vPos;
            colVel = (e69 - (e70 - e71));
        }
        let e74: vec2<f32> = pos;
        let e75: vec2<f32> = vPos;
        let e78: f32 = params.rule3Distance;
        if ((distance(e74, e75) < e78)) {
            let e80: vec2<f32> = cVel;
            let e81: vec2<f32> = vel;
            cVel = (e80 + e81);
            let e83: i32 = cVelCount;
            cVelCount = (e83 + 1);
        }
        continuing {
            let e86: u32 = i;
            i = (e86 + 1u);
        }
    }
    let e89: i32 = cMassCount;
    if ((e89 > 0)) {
        let e92: vec2<f32> = cMass;
        let e93: i32 = cMassCount;
        let e97: vec2<f32> = vPos;
        cMass = ((e92 / vec2<f32>(f32(e93))) - e97);
    }
    let e99: i32 = cVelCount;
    if ((e99 > 0)) {
        let e102: vec2<f32> = cVel;
        let e103: i32 = cVelCount;
        cVel = (e102 / vec2<f32>(f32(e103)));
    }
    let e107: vec2<f32> = vVel;
    let e108: vec2<f32> = cMass;
    let e110: f32 = params.rule1Scale;
    let e113: vec2<f32> = colVel;
    let e115: f32 = params.rule2Scale;
    let e118: vec2<f32> = cVel;
    let e120: f32 = params.rule3Scale;
    vVel = (((e107 + (e108 * e110)) + (e113 * e115)) + (e118 * e120));
    let e123: vec2<f32> = vVel;
    let e125: vec2<f32> = vVel;
    vVel = (normalize(e123) * clamp(length(e125), 0.0, 0.10000000149011612));
    let e131: vec2<f32> = vPos;
    let e132: vec2<f32> = vVel;
    let e134: f32 = params.deltaT;
    vPos = (e131 + (e132 * e134));
    let e138: f32 = vPos.x;
    if ((e138 < -1.0)) {
        vPos.x = 1.0;
    }
    let e144: f32 = vPos.x;
    if ((e144 > 1.0)) {
        vPos.x = -1.0;
    }
    let e150: f32 = vPos.y;
    if ((e150 < -1.0)) {
        vPos.y = 1.0;
    }
    let e156: f32 = vPos.y;
    if ((e156 > 1.0)) {
        vPos.y = -1.0;
    }
    let e164: vec2<f32> = vPos;
    particlesDst.particles[index].pos = e164;
    let e168: vec2<f32> = vVel;
    particlesDst.particles[index].vel = e168;
    return;
}
