@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32, 2>;

@compute @workgroup_size(1)
fn main() {
    result[0] += 1u;
    result[1] += u32(textureLoad(image, vec2<i32>(0), 0).r);
}

@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(positions[index], 0.0, 1.0);
}

@fragment
fn fragment() -> @location(0) vec4f {
    return textureLoad(image, vec2<i32>(0), 0);
}
