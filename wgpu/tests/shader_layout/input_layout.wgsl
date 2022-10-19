struct InputStruct {
    {{input_members}}
}

@group(0) @binding(0)
var<{{storage_type}}> input: InputStruct; 

@group(0) @binding(1)
var<storage, read_write> output: array<{{output_type}}>;

@compute @workgroup_size(1)
fn cs_main() {
    var i = 0u;
    {{body}}
}
