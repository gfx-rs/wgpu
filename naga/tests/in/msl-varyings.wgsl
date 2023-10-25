struct Vertex {
    @location(0) position: vec2f
}

struct NoteInstance {
    @location(1) position: vec2f
}

struct VertexOutput {
    @builtin(position) position: vec4f
}

@vertex
fn vs_main(vertex: Vertex, note: NoteInstance) -> VertexOutput {
    var out: VertexOutput;
    return out;
}

@fragment
fn fs_main(in: VertexOutput, note: NoteInstance) -> @location(0) vec4f {
    let position = vec3(1f);
    return in.position;
}
