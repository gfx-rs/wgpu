@group(0) @binding(0)
var texture: texture_2d_array<{{type}}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{type}}>;

@compute @workgroup_size(1)
fn copy_texture_to_buffer() {
    let layers = textureNumLayers(texture);
    let dim = textureDimensions(texture);
    for (var l = 0u; l < layers; l++) {
        for (var y = 0u; y < dim.y; y++) {
            for (var x = 0u; x < dim.x; x++) {
                output[(l * dim.y + y) * dim.x + x] = textureLoad(texture, vec2(x, y), l, 0).x;
            }
        }
    }
}
