// Hit object built-ins reject pointers to anything but a hit object.

enable wgpu_ray_query;
enable wgpu_ray_tracing_invocation_reorder;

@compute @workgroup_size(1)
fn main() {
    var rq: ray_query;
    _ = hitObjectIsHit(&rq);
}
