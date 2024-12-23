# big-compute-buffers

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

This example also assumes you've specifically come here looking to do this, because you want at least the following:
1. To be working on your 'data' in your shader treating it contiguously, not batching etc.
2. The data you are wanting to work on does **not** fit within a single buffer on your device, see the [hello](https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello) example for how to print information about your unique device to explore its maximum supported buffer size.

Demonstrates how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contiguous buffer on the GPU. This is known as 'chunking' or, sometimes 'pagination', not to be confused with the web kind, although they're not entirely dissimilar.

- Creates a large set of buffers totalling `1GB`, full of `0.0f32`.
- Moves those buffers to the DEVICE.
- Increments each element in said large buffer by `1.0`, on the DEVICE.
- Returns those modified `1.0` values as a back to the HOST.

This example uses the constants below, which are from the values that `wgpu` is guaranteed to support, _your_ hardware may support values higher than this.
>`src/big_compute_buffers/mod.rs`
```rust
const MAX_BUFFER_SIZE: u64 = 1 << 27; // 134_217_728 // 134MB
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1; // 65_535
```

## Caution:
- Large buffers can fail to allocate due to fragmentation issues, you will **always** need not only the appropriate amount of space required for your buffer(s) but, that space will also need to be contiguous within GPU/Device memory for this strategy to work. You can read more about fragmentation [here](https://developer.nvidia.com/docs/drive/drive-os/archives/6.0.4/linux/sdk/common/topics/graphics_content/avoiding_memory_fragmentation.html).
- `wgsl` is, compared to the GP-GPU compute workhorse languages like CUDA very immature.

## To Run
```sh
RUST_LOG=wgpu_examples::big_compute_buffers=info cargo run -r --bin wgpu-examples -- big_compute_buffers
```

## Example Output
```
[2024-09-29T11:47:55Z INFO  wgpu_examples::big_compute_buffers] All 0.0s
[2024-09-29T11:47:58Z INFO  wgpu_examples::big_compute_buffers] GPU RUNTIME: 3228ms
[2024-09-29T11:47:58Z INFO  wgpu_examples::big_compute_buffers] All 1.0s
```
