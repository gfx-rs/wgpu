# big-compute-buffers

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

Showcases how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contiguous buffer on the GPU.

- Creates a large buffer, by default 1GB, full of `0.0`s.
- Increments each element in said large buffer by `1.0`

## To Run
As the maximum supported buffer size varies wildly per system, when you try to run this, then when it will likely fail, in-which-case read the error and update these `const`s accordingly:
>`src/big_compute_buffers/mod.rs`
```rust
const MAX_BUFFER_SIZE: u64 = 1 << 27; // 134_217_728 // 134MB
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1; // 65_535
```

It is recommended you enable the logger to see the code explain what it's doing.
```
 RUST_LOG=wgpu_examples::big_compute_buffers cargo run -r --bin wgpu-examples big_compute_buffers
```

## Example Output
<detail>
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
</detail>

## FAQ

### How do I ascertain the max_*buffer_binding_size?
You could write some code, or just run this example and it'll fail then see [below](#how-will-i-know-if-the-provided-default-is-too-high-for-my-gpu).
```rust
// get yourself a wgpu::Device
device.limits().max_buffer_size
````

### How will I know if the provided default is too high for my gpu?
You'll see this error:
```sh
thread 'main' panicked at wgpu/src/backend/wgpu_core.rs:3433:5:
wgpu error: Validation Error

Caused by:
  In Device::create_bind_group, label = 'Combined Storage Bind Group'
    Buffer binding 0 range 268435456 exceeds `max_*_buffer_binding_size` limit 134217728
```


### What's going on with the MAX_DISPATCH_SIZE, and the OFFSET etc in the shader?
There is a limit to the maximum number of invocations you can spool up in `workgroup_size(x,y,z)`, this means there's a limit, on how 'high' a number we can get a global_index.xyz value to count up to, if this limit is less than the number of `0.0`s in our buffers we need a way to count 'higher'. 
