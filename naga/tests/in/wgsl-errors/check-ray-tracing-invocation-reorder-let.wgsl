// A `let` binding of a hit object is rejected, since a `let` requires an
// initializer.

enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;

@ray_generation
fn main() {
    var ho: hit_object;
    let _ho_1 = ho;
}
