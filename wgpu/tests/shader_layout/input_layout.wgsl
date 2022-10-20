struct InputStruct {
    {{input_members}}
}

// The following removed by code in push contant case
@group(0) @binding(0)
var<{{storage_type}}> input: InputStruct; 

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn cs_main() {
    let loaded = input;
    var i = 0u;
    {{body}}
}
