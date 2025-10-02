// Compiled with:
// slangc -target spirv -profile spirv_1_5 -o naga/tests/in/spv/gather-cmp.spv naga/tests/in/spv/gather-cmp.slang
// Disassembled with:
// spirv-dis naga/tests/in/spv/gather-cmp.spv -o naga/tests/in/spv/gather-cmp.spvasm

#language slang 2026

[[vk::binding(0, 0)]] var texture: Texture2D;
[[vk::binding(1, 0)]] var depth_sampler: SamplerComparisonState;

struct VertexOutput {
    var texture_coordinates: float2;
};

[[shader("pixel")]]
func main(input: VertexOutput) -> float4 {
    return texture.GatherCmp(depth_sampler, input.texture_coordinates, 0.5);
}
