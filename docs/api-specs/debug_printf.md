# Shader `debugPrintf`

`wgpu` supports shader debug printing on native backends when `Features::DEBUG_PRINTF` is enabled.
This is a debugging extension and is not part of core WebGPU.

## Requirements

- Request `Features::DEBUG_PRINTF` when creating the device.
- Add `enable wgpu_debug_printf;` to each WGSL module that calls `debugPrintf`.
- Use a backend that advertises the feature:
  - Metal with shader logging support, available in Metal 3.2 and later.
  - Vulkan with `VK_KHR_shader_non_semantic_info` support.

On Vulkan, `debugPrintf` output is produced through the validation layer debug-printf path. It is not enabled when GPU-assisted validation is enabled, because the two validation features are mutually exclusive.

## WGSL Syntax

`debugPrintf` is a statement-like built-in:

```wgsl
enable wgpu_debug_printf;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    debugPrintf("invocation: %u %u %u", id.x, id.y, id.z);
}
```

The first argument must be a string literal. String literals are currently only accepted as the format argument to `debugPrintf`.

Remaining arguments must currently be scalar values. Vector and matrix arguments may be supported in the future, but for now vector components should be passed individually.

Format string interpretation follows the active backend's shader logging implementation. The supported format syntax is therefore intentionally limited to the common C-style debug printf forms accepted by Metal shader logging and Vulkan shader debug printf.

## Backend Notes

- Metal lowers `debugPrintf` to `metal::os_log_default.log_info`.
- Vulkan lowers `debugPrintf` through SPIR-V `NonSemantic.DebugPrintf`.

## References

- Apple Metal shader logging: https://developer.apple.com/documentation/metal/logging-shader-debug-messages
- Vulkan shader debug printf sample: https://docs.vulkan.org/samples/latest/samples/extensions/shader_debugprintf/README.html
