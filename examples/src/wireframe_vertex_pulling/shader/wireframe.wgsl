struct Uniforms {
    world           : mat4x4<f32>,
    view            : mat4x4<f32>,
    proj            : mat4x4<f32>,
    screen_width    : f32,
    screen_height   : f32,
    _padding        : vec2<f32>,
}

struct U32s {
    values : array<u32>
}

struct F32s {
    values : array<f32>
}

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<storage, read> positions : F32s;
@binding(2) @group(0) var<storage, read> colors : U32s;
@binding(3) @group(0) var<storage, read> indices : U32s;

struct VertexInput {
    @builtin(instance_index) instanceID : u32,
    @builtin(vertex_index) vertexID : u32
}

struct VertexOutput {
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec4<f32>
}

@vertex
fn main_vertex(vertex : VertexInput) -> VertexOutput {
    var localToElement = array<u32, 6>(0u, 1u, 1u, 2u, 2u, 0u);

    var triangleIndex = vertex.vertexID / 6u;
    var localVertexIndex = vertex.vertexID % 6u;

    var elementIndexIndex = 3u * triangleIndex + localToElement[localVertexIndex];
    var elementIndex = indices.values[elementIndexIndex];

    var position = vec4<f32>(
        positions.values[3u * elementIndex + 0u],
        positions.values[3u * elementIndex + 1u],
        positions.values[3u * elementIndex + 2u],
        1.0
    );

    position = uniforms.proj * uniforms.view * uniforms.world * position;

    var color_u32 = colors.values[elementIndex];
    var color = vec4<f32>(
        f32((color_u32 >>  0u) & 0xFFu) / 255.0,
        f32((color_u32 >>  8u) & 0xFFu) / 255.0,
        f32((color_u32 >> 16u) & 0xFFu) / 255.0,
        f32((color_u32 >> 24u) & 0xFFu) / 255.0,
    );

    var output : VertexOutput;
    output.position = position;
    output.color = color;

    return output;
}

struct FragmentInput {
    @location(0) color : vec4<f32>
}

struct FragmentOutput {
    @location(0) color : vec4<f32>
}

@fragment
fn main_fragment(fragment : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = fragment.color;

    return output;
}