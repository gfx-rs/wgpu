// Some credit to https://github.com/paulgb/wgsl-playground/tree/main.

// We use seperate the x and y instead of using a vec2 to avoid wgsl padding.
struct AppState {
    pos_x: f32,
    pos_y: f32,
    zoom: f32,
    max_iterations: u32,
}

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) coord: vec2<f32>,
};

@group(0)
@binding(0)
var<uniform> app_state: AppState;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var vertices = array<vec2<f32>, 3>(
        vec2<f32>(-1., 1.),
        vec2<f32>(3.0, 1.),
        vec2<f32>(-1., -3.0),
    );
    var out: VertexOutput;
    out.coord = vertices[in.vertex_index];
    out.position = vec4<f32>(out.coord, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let max_iterations = app_state.max_iterations;
    var final_iteration = max_iterations;
    let c = vec2(
        // Translated to put everything nicely in frame.
        (in.coord.x) * 3.0 / app_state.zoom + app_state.pos_x,
        (in.coord.y) * 3.0 / app_state.zoom + app_state.pos_y
    );
    var current_z = c;
    var next_z: vec2<f32>;
    for (var i = 0u; i < max_iterations; i++) {
        next_z.x = (current_z.x * current_z.x - current_z.y * current_z.y) + c.x;
        next_z.y = (2.0 * current_z.x * current_z.y) + c.y;
        current_z = next_z;
        if length(current_z) > 4.0 {
            final_iteration = i;
            break;
        }
    }
    let value = f32(final_iteration) / f32(max_iterations);

    return vec4(value, value, value, 1.0);
}