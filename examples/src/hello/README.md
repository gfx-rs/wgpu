# hello

This example prints output describing the adapter in use.

## To Run

```
cargo run --bin wgpu-examples hello
```

## Example output

```
# You might see different output as it depends on your graphics card and drivers
Available adapters:
    AdapterInfo { name: "AMD RADV VEGA10", vendor: 4098, device: 26751, device_type: DiscreteGpu, backend: Vulkan }
    AdapterInfo { name: "llvmpipe (LLVM 12.0.0, 256 bits)", vendor: 65541, device: 0, device_type: Cpu, backend: Vulkan }
    AdapterInfo { name: "Radeon RX Vega (VEGA10, DRM 3.41.0, 5.13.0-52-generic, LLVM 12.0.0)", vendor: 4098, device: 0, device_type: Other, backend: Gl }
Selected adapter: AdapterInfo { name: "AMD RADV VEGA10", vendor: 4098, device: 26751, device_type: DiscreteGpu, backend: Vulkan }
```
