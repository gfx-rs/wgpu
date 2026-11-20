@group(0) @binding(0)
var inputTexture: texture_storage_3d<rgba8unorm, read>;
@group(0) @binding(1)
var outputTexture: texture_storage_3d<rgba8unorm, write>;

@fragment
fn main() {
    let srcSize = textureDimensions(inputTexture);
    let dstSize = textureDimensions(outputTexture);
}
