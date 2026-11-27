/*!
# Coherent and volatile buffers in shaders

On some backends, `wgpu` supports additional `@coherent` and `@volatile` attributes applied to storage buffers in WGSL shaders.
Using these attributes requires [`Features::MEMORY_DECORATION_COHERENT`] and [`Features::MEMORY_DECORATION_VOLATILE`], respectively.

These attributes affect the memory behavior of naga-compiled shaders.

***This is not a core part of WebGPU, but is a wgpu-specific extension.***

## WebGPU defined behavior

[WebGPU follows the Vulkan Memory Model](https://gpuweb.github.io/gpuweb/wgsl/#memory-model) for the most part.

## wgpu default behavior

## Coherent behavior

## Volatile behavior

*/

use crate::Features;
