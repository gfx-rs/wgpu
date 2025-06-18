enable clip_distances;
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @builtin(clip_distances) clip_distances: array<f32, 1>,
}

@vertex
fn main() -> VertexOutput {
    var out: VertexOutput;
    out.clip_distances[0] = 0.5;
    return out;
}