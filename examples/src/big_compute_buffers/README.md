# WIP


# big-compute-buffers

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

Showcases how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contiguous buffer on the GPU.

- Creates a large set of buffers totalling `1GB`, full of `0.0f32`.
- Increments each element in said large buffer by `1.0`.
- Returns those `1.0` values as a back to the HOST.

## To Run
As the maximum supported buffer size varies wildly per system, you _should_ check the output of `wgpu::Limits` to ensure that the amount of data _you_ have defnitely doesn't fit into a single buffer.


>`src/big_compute_buffers/mod.rs`
```rust
const MAX_BUFFER_SIZE: u64 = 1 << 27; // 134_217_728 // 134MB
const MAX_DISPATCH_SIZE: u32 = (1 << 16) - 1; // 65_535
```

```
 cargo run -r --bin wgpu-examples big_compute_buffers
```

## Example Output

<details>
  <summary>Output</summary>

```sh
[DEBUG wgpu_examples::big_compute_buffers] GPU RUNTIME: 2522ms
```

</details>

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
