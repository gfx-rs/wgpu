# Copyright 2020 The Tint Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import "GLSL.std.450" as std;

# vertex shader

[[location 0]] var<in> a_particlePos : vec2<f32>;
[[location 1]] var<in> a_particleVel : vec2<f32>;
[[location 2]] var<in> a_pos : vec2<f32>;
[[builtin position]] var gl_Position : vec4<f32>;

fn vtx_main() -> void {
  var angle : f32 = -std::atan2(a_particleVel.x, a_particleVel.y);
  var pos : vec2<f32> = vec2<f32>(
      (a_pos.x * std::cos(angle)) - (a_pos.y * std::sin(angle)),
      (a_pos.x * std::sin(angle)) + (a_pos.y * std::cos(angle)));
  gl_Position = vec4<f32>(pos + a_particlePos, 0, 1);
  return;
}
entry_point vertex as "main" = vtx_main;

# fragment shader
[[location 0]] var<out> fragColor : vec4<f32>;

fn frag_main() -> void {
  fragColor = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  return;
}
entry_point fragment as "main" = frag_main;

# compute shader
type Particle = struct {
  [[offset 0]] pos : vec2<f32>;
  [[offset 8]] vel : vec2<f32>;
};

type SimParams = struct {
  [[offset 0]] deltaT : f32;
  [[offset 4]] rule1Distance : f32;
  [[offset 8]] rule2Distance : f32;
  [[offset 12]] rule3Distance : f32;
  [[offset 16]] rule1Scale : f32;
  [[offset 20]] rule2Scale : f32;
  [[offset 24]] rule3Scale : f32;
};

type Particles = struct {
  [[offset 0]] particles : array<Particle, 5>;
};

[[binding 0, set 0]] var<uniform> params : SimParams;
[[binding 1, set 0]] var<storage_buffer> particlesA : Particles;
[[binding 2, set 0]] var<storage_buffer> particlesB : Particles;

[[builtin global_invocation_id]] var gl_GlobalInvocationID : vec3<u32>;

# https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
fn compute_main() -> void {
  var index : u32 = gl_GlobalInvocationID.x;
  if (index >= 5) {
    return;
  }

  var vPos : vec2<f32> = particlesA.particles[index].pos;
  var vVel : vec2<f32> = particlesA.particles[index].vel;

  var cMass : vec2<f32> = vec2<f32>(0, 0);
  var cVel : vec2<f32> = vec2<f32>(0, 0);
  var colVel : vec2<f32> = vec2<f32>(0, 0);
  var cMassCount : i32 = 0;
  var cVelCount : i32 = 0;

  var pos : vec2<f32>;
  var vel : vec2<f32>;
  var i : u32 = 0;
  loop {
    if (i >= 5) {
      break;
    }
    if (i == index) {
      continue;
    }

    pos = particlesA.particles[i].pos.xy;
    vel = particlesA.particles[i].vel.xy;

    if (std::distance(pos, vPos) < params.rule1Distance) {
      cMass = cMass + pos;
      cMassCount = cMassCount + 1;
    }
    if (std::distance(pos, vPos) < params.rule2Distance) {
      colVel = colVel - (pos - vPos);
    }
    if (std::distance(pos, vPos) < params.rule3Distance) {
      cVel = cVel + vel;
      cVelCount = cVelCount + 1;
    }

    continuing {
      i = i + 1;
    }
  }
  if (cMassCount > 0) {
    cMass = (cMass / vec2<f32>(cMassCount, cMassCount)) + vPos;
  }
  if (cVelCount > 0) {
    cVel = cVel / vec2<f32>(cVelCount, cVelCount);
  }

  vVel = vVel + (cMass * params.rule1Scale) + (colVel * params.rule2Scale) +
      (cVel * params.rule3Scale);

  # clamp velocity for a more pleasing simulation
  vVel = std::normalize(vVel) * std::fclamp(std::length(vVel), 0.0, 0.1);

  # kinematic update
  vPos = vPos + (vVel * params.deltaT);

  # Wrap around boundary
  if (vPos.x < -1.0) {
    vPos.x = 1.0;
  }
  if (vPos.x > 1.0) {
    vPos.x = -1.0;
  }
  if (vPos.y < -1.0) {
    vPos.y = 1.0;
  }
  if (vPos.y > 1.0) {
    vPos.y = -1.0;
  }

  # Write back
  particlesB.particles[index].pos = vPos;
  particlesB.particles[index].vel = vVel;

  return;
}
entry_point compute as "main" = compute_main;

