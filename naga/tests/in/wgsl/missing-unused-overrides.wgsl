override ov_for_vertex: f32;

@vertex
fn vert_main(
  @location(0) pos : vec2<f32>,
  @builtin(instance_index) ii: u32,
  @builtin(vertex_index) vi: u32,
) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos.x * ov_for_vertex, pos.y, 0.0, 1.0);
}

struct FragmentIn {
    @location(0) color: vec4<f32>
}

override ov_for_fragment: f32;

fn frag_helper(color: vec4<f32>) -> vec4<f32> {
    return color * ov_for_fragment;
}

@fragment
fn frag_main(in: FragmentIn) -> @location(0) vec4<f32> {
    return frag_helper(in.color);
}

override ov_global_init: u32;
var<private> foo: u32 = ov_global_init;

override ov_array_size: u32;
var<workgroup> arr: array<u32, ov_array_size>;

override ov_for_compute: u32;

fn compute_helper() {
    _ = foo;
    _ = arr[0];
}

override ov_workgroup_size: u32;
@compute @workgroup_size(ov_workgroup_size)
fn compute_main() {
    _ = ov_for_compute;
    compute_helper();
}
