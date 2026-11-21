// `hitObjectRecordFromQuery` requires a pointer to a ray query, not to
// anything else.

enable wgpu_ray_tracing_pipeline;
enable wgpu_ray_tracing_invocation_reorder;

@ray_generation
fn main() {
    var ho: hit_object;
    hitObjectRecordFromQuery(&ho, &ho);
}
