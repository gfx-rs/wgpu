// The operands of `reorderThread` are `u32`, so anything that does not
// automatically convert to `u32` is rejected.

enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;

@ray_generation
fn main() {
    reorderThread(1.0, 2u);
}
