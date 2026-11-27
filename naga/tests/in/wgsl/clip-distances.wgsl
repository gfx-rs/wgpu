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

struct VertexOutputSplit {
    @location(0) color: vec4<f32>,
    @builtin(clip_distances) clip_distances: array<f32, 6>,
    @builtin(position) position: vec4<f32>,
}

@vertex
fn main_split() -> VertexOutputSplit {
    var out: VertexOutputSplit;
    for (var i = 0; i < 6; i++) {
        out.clip_distances[i] = f32(i) - 2.5;
    }
    return out;
}
