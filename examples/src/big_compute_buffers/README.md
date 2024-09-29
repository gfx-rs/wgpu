# WIP


# big-compute-buffers

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

Showcases how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contiguous buffer on the GPU.

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

## To Run
```
RUST_LOG=wgpu_examples::big_compute_buffers=info cargo run -r --bin wgpu-examples -- big_compute_buffers
```

## Example Output
```sh
[2024-09-29T11:32:44Z INFO  wgpu_examples::big_compute_buffers] All 0.0s
[2024-09-29T11:32:47Z INFO  wgpu_examples::big_compute_buffers] GPU RUNTIME: 3312ms
[2024-09-29T11:32:47Z INFO  wgpu_examples::big_compute_buffers] All 1.0s
```
