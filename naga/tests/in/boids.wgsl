const NUM_PARTICLES: u32 = 1500u;

struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
}

struct SimParams {
  deltaT : f32,
  rule1Distance : f32,
  rule2Distance : f32,
  rule3Distance : f32,
  rule1Scale : f32,
  rule2Scale : f32,
  rule3Scale : f32,
}

struct Particles {
  particles : array<Particle>
}

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage> particlesSrc : Particles;
@group(0) @binding(2) var<storage,read_write> particlesDst : Particles;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
  let index : u32 = global_invocation_id.x;
  if index >= NUM_PARTICLES {
    return;
  }

  var vPos = particlesSrc.particles[index].pos;
  var vVel = particlesSrc.particles[index].vel;

  var cMass = vec2<f32>(0.0, 0.0);
  var cVel = vec2<f32>(0.0, 0.0);
  var colVel = vec2<f32>(0.0, 0.0);
  var cMassCount : i32 = 0;
  var cVelCount : i32 = 0;

  var pos : vec2<f32>;
  var vel : vec2<f32>;
  var i : u32 = 0u;
  loop {
    if i >= NUM_PARTICLES {
      break;
    }
    if i == index {
      continue;
    }

    pos = particlesSrc.particles[i].pos;
    vel = particlesSrc.particles[i].vel;

    if distance(pos, vPos) < params.rule1Distance {
      cMass = cMass + pos;
      cMassCount = cMassCount + 1;
    }
    if distance(pos, vPos) < params.rule2Distance {
      colVel = colVel - (pos - vPos);
    }
    if distance(pos, vPos) < params.rule3Distance {
      cVel = cVel + vel;
      cVelCount = cVelCount + 1;
    }

    continuing {
      i = i + 1u;
    }
  }
  if cMassCount > 0 {
    cMass = cMass / f32(cMassCount) - vPos;
  }
  if cVelCount > 0 {
    cVel = cVel / f32(cVelCount);
  }

  vVel = vVel + (cMass * params.rule1Scale) +
      (colVel * params.rule2Scale) +
      (cVel * params.rule3Scale);

  // clamp velocity for a more pleasing simulation
  vVel = normalize(vVel) * clamp(length(vVel), 0.0, 0.1);

  // kinematic update
  vPos = vPos + (vVel * params.deltaT);

  // Wrap around boundary
  if vPos.x < -1.0 {
    vPos.x = 1.0;
  }
  if vPos.x > 1.0 {
    vPos.x = -1.0;
  }
  if vPos.y < -1.0 {
    vPos.y = 1.0;
  }
  if vPos.y > 1.0 {
    vPos.y = -1.0;
  }

  // Write back
  particlesDst.particles[index].pos = vPos;
  particlesDst.particles[index].vel = vVel;
}
