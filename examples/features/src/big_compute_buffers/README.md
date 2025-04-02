# big-compute-buffers

*NOTE: `binding_array` is Vulkan only.*

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

This example also assumes you've specifically come here looking to do this, because you want at least the following:

1. To be working on your 'data' in your shader treating it contiguously, not batching etc.
2. The data you are wanting to work on does **not** fit within a single buffer on your device, see the [hello](https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello) example for how to print information about your unique device to explore its maximum supported buffer size.

Demonstrates how to split larger datasets (things too big to fit into a single buffer), across multiple buffers.

- Creates a set of buffers totalling `1GB`, full of `0.0f32`.
- Moves those buffers to the DEVICE.
- Increments each element in each set of buffers by `1.0`, on the DEVICE.
- Returns those modified buffers full of `1.0` values as a back to the HOST.

## Caution

- Large buffers can fail to allocate due to fragmentation issues, you will **always** need not only the appropriate amount of space required for your buffer(s) but, that space will also need to be contiguous within GPU/Device memory for this strategy to work.

You can read more about fragmentation [here](https://developer.nvidia.com/docs/drive/drive-os/archives/6.0.4/linux/sdk/common/topics/graphics_content/avoiding_memory_fragmentation.html).

## To Run

```sh
# linux/mac
RUST_LOG=wgpu_examples::big_compute_buffers=info cargo run -r --bin wgpu-examples -- big_compute_buffers

# windows (Powershell)
$env:WGPU_BACKEND="Vulkan"; $env:RUST_LOG="wgpu_examples::big_compute_buffers=info"; cargo run -r --bin wgpu-examples -- big_compute_buffers
```

## Example Output

```txt
[2024-09-29T11:47:55Z INFO  wgpu_examples::big_compute_buffers] All 0.0s
[2024-09-29T11:47:58Z INFO  wgpu_examples::big_compute_buffers] GPU RUNTIME: 3228ms
[2024-09-29T11:47:58Z INFO  wgpu_examples::big_compute_buffers] All 1.0s
```
