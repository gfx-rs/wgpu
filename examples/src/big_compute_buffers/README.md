# big-compute-buffers

Showcases how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contigious buffer on the GPU.

- Creates a large buffer full of `0.0`s.
- Increments each element in said large buffer by `1.0`


## To Run

As the maximum supported buffer size varies wildly per system, you try to run this, then when it likely fails update the two constants appropriately:
```rust
const MAX_BUFFER_SIZE: u64 = 1 << 27; // 34_217_728
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1; // 65_535
```

```
RUST_LOG=big_compute_buffers cargo run --bin wgpu-examples big_compute_buffers
```

## Example Output

```
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] Size of input 1_073_741_824b
[2024-08-20T20:44:49Z WARN  wgpu_examples::big_compute_buffers] Supplied input is too large for a single staging buffer, splitting...
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] num_chunks: 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 1 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 2 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 3 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 4 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 5 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 6 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 7 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating staging buffer 8 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] Created staging_buffer
[2024-08-20T20:44:49Z WARN  wgpu_examples::big_compute_buffers] Supplied input is too large for a single storage buffer, splitting...
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 1 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 2 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 3 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 4 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 5 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 6 of 8
[2024-08-20T20:44:49Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 7 of 8
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] creating Storage Buffer 8 of 8
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] Created storage_buffer
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:0 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:1 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:2 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:3 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:4 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:5 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:6 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] bind_idx:7 buffer is 134_217_728b
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] created 8 BindGroupEntries with 8 corresponding BindGroupEntryLayouts.
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] set_pipeline complete
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] set_bind_group complete
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] buffers created, submitting job to GPU
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] Job submission complete.
[2024-08-20T20:44:50Z DEBUG wgpu_examples::big_compute_buffers] Getting results...
[2024-08-20T20:44:57Z DEBUG wgpu_examples::big_compute_buffers] GPU RUNTIME: 2522ms
[2024-08-20T20:44:59Z DEBUG wgpu_examples::big_compute_buffers] All numbers checked, previously 0.0 elements are now 1.0s
```
