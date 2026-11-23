// `first_vertex` and `first_instance` are added once, however the built-ins are used.

struct Input {
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}

fn sum_indices(in: Input) -> u32 {
    return in.vertex_index + in.instance_index;
}

@vertex
fn copy_struct_members(in: Input) -> @builtin(position) vec4<f32> {
    let vertex = in.vertex_index;
    let instance = in.instance_index;
    return vec4<f32>(f32(vertex), f32(instance), 0.0, 1.0);
}

@vertex
fn pass_struct_to_function(in: Input) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(sum_indices(in)), 0.0, 0.0, 1.0);
}

@vertex
fn copy_parameters(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> @builtin(position) vec4<f32> {
    let vertex = vertex_index;
    let instance = instance_index;
    return vec4<f32>(f32(vertex), f32(instance), 0.0, 1.0);
}
