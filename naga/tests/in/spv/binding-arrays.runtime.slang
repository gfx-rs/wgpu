// Compiled with:
// slangc -target spirv -profile spirv_1_5 -o naga/tests/in/spv/binding-arrays.runtime.spv naga/tests/in/spv/binding-arrays.runtime.slang
// Disassembled with:
// spirv-dis naga/tests/in/spv/binding-arrays.runtime.spv -o naga/tests/in/spv/binding-arrays.runtime.spvasm

#language slang 2026

[[vk::binding(0, 0)]] var textures: Texture2D[];
[[vk::binding(1, 0)]] var linear_sampler: SamplerState;

struct VertexOutput {
    var texture_coordinates: float2;
    var texture_index: uint;
};

[[shader("pixel")]]
func main(input: VertexOutput) -> float4 {
    return textures[input.texture_index].Sample(linear_sampler, input.texture_coordinates);
}
