@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main()
{
    for (var i = 0u; i < 4; i++)
    {
        data[i] = i * 2; // Example operation: double each element
    }
}
