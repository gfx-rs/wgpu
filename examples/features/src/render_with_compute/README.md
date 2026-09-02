# render_with_compute

This example draws every frame from a compute shader instead of a render
pipeline. The compute shader writes into a storage texture, which is then
blitted to the surface.

Firefox caps this at about 10 FPS on WebGPU because of
<https://bugzilla.mozilla.org/show_bug.cgi?id=1870699>, so this is a
demonstration rather than a pattern to copy.

## To Run

```
cargo run --bin wgpu-examples render_with_compute
```
