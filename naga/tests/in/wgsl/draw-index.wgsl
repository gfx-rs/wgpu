enable draw_index;

struct Input {
    @builtin(draw_index) draw_index: u32,
}

@vertex
fn vertex(input: Input) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(input.draw_index), 1.0, 1.0, 1.0);
}