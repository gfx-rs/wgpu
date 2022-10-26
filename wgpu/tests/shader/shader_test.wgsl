struct InputStruct {
    {{input_members}}
}

{{input_bindings}}
var<{{storage_type}}> input: InputStruct; 

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
    {{body}}
}
