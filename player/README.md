# wgpu player

This is an application that allows replaying the `wgpu` workloads recorded elsewhere. It requires the player to be built from
the same revision as an application was linking to, or otherwise the data may fail to load.

Launch as:
```rust
play <trace-dir>
```

When built with "winit" feature, it's able to replay the workloads that operate on a swapchain. It renders each frame sequentially, then waits for the user to close the window. When built without "winit", it launches in console mode and can replay any trace that doesn't use swapchains.

Note: replaying is currently restricted to the same backend, as one used for recording a trace. It is straightforward, however, to just replace the backend in RON, since it's serialized as plain text. Valid values are: Vulkan, Metal, Dx12, and Dx11.
