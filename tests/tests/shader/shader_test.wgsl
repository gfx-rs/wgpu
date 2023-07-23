struct CustomStruct {
    {{input_members}}
}

{{input_bindings}}
var<{{storage_type}}> input: {{input_type}}; 

@group(0) @binding(1)
var<storage, read_write> output: {{output_type}};

@compute @workgroup_size(1)
fn cs_main() {
    {{body}}
}
