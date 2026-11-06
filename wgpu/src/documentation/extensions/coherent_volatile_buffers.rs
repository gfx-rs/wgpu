/*!
# Coherent and volatile buffers in shaders

On some backends, `wgpu` supports additional `@coherent` and `@volatile` attributes applied to storage buffers in WGSL shaders.
Using these attributes requires [`Features::MEMORY_DECORATION_COHERENT`] and [`Features::MEMORY_DECORATION_VOLATILE`], respectively.

***This is not a core part of WebGPU, but is a wgpu-specific extension.***

## WebGPU defined behavior

## wgpu default behavior

## Coherent behavior

## Volatile behavior

*/

use crate::Features;
