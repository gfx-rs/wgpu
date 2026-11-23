enable f16;

@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f16>;
@group(0) @binding(2) var<storage, read_write> vecs: array<vec2<f16>>;

@compute @workgroup_size(1)
fn direct(@builtin(global_invocation_id) id: vec3<u32>) {
    outputs[id.x] = f16(inputs[id.x]);
}

@compute @workgroup_size(1)
fn direct_vec(@builtin(global_invocation_id) id: vec3<u32>) {
    vecs[id.x] = vec2<f16>(vec2<f32>(inputs[id.x], inputs[id.x + 1u]));
}

@compute @workgroup_size(1)
fn reused() {
    let h = f16(inputs[0]);
    outputs[0] = h;
    outputs[1] = h;
}

@compute @workgroup_size(1)
fn arithmetic() {
    outputs[0] = f16(inputs[0]) + f16(1.0);
}

@compute @workgroup_size(1)
fn local_store() {
    var h = f16(inputs[0]);
    outputs[0] = h;
}
