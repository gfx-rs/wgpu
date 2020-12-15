[[location(0)]]
var<in> in_particle_pos: vec2<f32>;
[[location(1)]]
var<in> in_particle_vel: vec2<f32>;
[[location(2)]]
var<in> in_position: vec2<f32>;
[[builtin(position)]]
var<out> out_position: vec4<f32>;

[[stage(vertex)]]
fn main() {
    const angle: f32 = -atan2(in_particle_vel.x, in_particle_vel.y);
    const pos: vec2<f32> = vec2<f32>(
        in_position.x * cos(angle) - in_position.y * sin(angle),
        in_position.x * sin(angle) + in_position.y * cos(angle)
    );
    out_position = vec4<f32>(pos + in_particle_pos, 0.0, 1.0);
}

[[location(0)]]
var<out> out_color: vec4<f32>;

[[stage(fragment)]]
fn main() {
    out_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
