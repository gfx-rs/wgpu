struct VertexInput {
    @location(0) chunk : vec3<i32>,
    @location(1) texture_index : u32,
}

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
};

@vertex
fn vs_main(
in : VertexInput,
) -> VertexOutput {
    var out : VertexOutput;

    let position = vec3<f32> (in.chunk - vec3<i32>(5));

    return out;
}
