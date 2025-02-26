var<workgroup> sized_comma: array<u32, 1,>;
var<workgroup> sized_no_comma: array<u32, 1>;

@group(0) @binding(0)
var<storage, read_write> unsized_comma: array<u32,>;

@group(0) @binding(1)
var<storage, read_write> unsized_no_comma: array<u32>;

@compute @workgroup_size(1)
fn main() {
    sized_comma[0] = unsized_comma[0];
    sized_no_comma[0] = unsized_no_comma[0];
    unsized_no_comma[0] = sized_comma[0] + sized_no_comma[0];
}
