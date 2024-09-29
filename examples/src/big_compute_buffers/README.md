# big-compute-buffers

This example assumes you're familiar with the other GP-GPU compute examples in this repository, if you're not you should go look at those first.

Showcases how to split larger datasets (things too big to fit into a single buffer), across multiple buffers whilst treating them as a single, contiguous buffer on the GPU.

- Creates a large set of buffers totalling `1GB`, full of `0.0f32`.
- Increments each element in said large buffer by `1.0`.
- Returns those `1.0` values as a back to the HOST.

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

<details>
  <summary>Output</summary>

```sh
[DEBUG wgpu_examples::big_compute_buffers] GPU RUNTIME: 2522ms
```

</details>
 w
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
