// Compiled with:
// slangc -target spirv -profile spirv_1_5 -o naga/tests/in/spv/load-ms-texture.spv naga/tests/in/spv/load-ms-texture.slang
// Disassembled with:
// spirv-dis naga/tests/in/spv/load-ms-texture.spv -o naga/tests/in/spv/load-ms-texture.spvasm
#language slang 2026

[[vk::binding(0, 0)]] var texture: Texture2DMS;

[[shader("pixel")]]
func fs_main(float4 position : SV_Position) -> float4 {
    let pixel_coord = int2(position.xy);
    var color: float4;

    for (var index: int = 0; index < 8; index++) {
        color += texture.Load(pixel_coord, index);
    }

    color = color / 8.0;

    return color;
}
