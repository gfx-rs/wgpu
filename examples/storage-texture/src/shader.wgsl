const MAX_ITERATIONS: u32 = 50u;

@group(0)
@binding(0)
var texture: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var final_iteration = MAX_ITERATIONS;
    var c = vec2(
        // Translated to put everything nicely in frame.
        (f32(id.x) / f32(textureDimensions(texture).x)) * 3.0 - 2.25,
        (f32(id.y) / f32(textureDimensions(texture).y)) * 3.0 - 1.5
    );
    var current_z = c;
    var next_z: vec2<f32>;
    for (var i = 0u; i < MAX_ITERATIONS; i++) {
        next_z.x = (current_z.x * current_z.x - current_z.y * current_z.y) + c.x;
        next_z.y = (2.0 * current_z.x * current_z.y) + c.y;
        current_z = next_z;
        if length(current_z) > 4.0 {
            final_iteration = i;
            break;
        }
    }
    let value = f32(final_iteration) / f32(MAX_ITERATIONS);
    textureStore(texture, vec2(i32(id.x), i32(id.y)), vec4(value, value, value, 1.0));
}