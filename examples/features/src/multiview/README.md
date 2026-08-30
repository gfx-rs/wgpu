# multiview

This example renders different content to different layers of an array texture
in one draw call, using multiview. VR applications use this to render both eyes
at once. The layer mask in the example is non-contiguous, to show that the
layers a multiview pass writes need not be adjacent.

## To Run

```
cargo run --bin wgpu-examples multiview
```
