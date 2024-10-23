# big-compute-buffers

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

Demonstrates how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contiguous buffer on the GPU.

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

Note:
- Large buffers can fail to allocate due to fragmentation issues.
- Working with large buffers is not always ideal, you should also see the pagination example to see if that approach suits your needs better.

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
