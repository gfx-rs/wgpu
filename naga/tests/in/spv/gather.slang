// Compiled with:
// slangc -target spirv -profile spirv_1_5 -o naga/tests/in/spv/gather.spv naga/tests/in/spv/gather.slang
// Disassembled with:
// spirv-dis naga/tests/in/spv/gather.spv -o naga/tests/in/spv/gather.spvasm

#language slang 2026

[[vk::binding(0, 0)]] var texture: Texture2D;
[[vk::binding(1, 0)]] var linear_sampler: SamplerState;

struct VertexOutput {
    var texture_coordinates: float2;
};

[[shader("pixel")]]
func main(input: VertexOutput) -> float4 {
    return texture.GatherGreen(linear_sampler, input.texture_coordinates);
}
