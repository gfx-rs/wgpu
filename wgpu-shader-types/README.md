# wgpu-shader-types

wgpu-shader-types contains some types used by both naga and wgpu-shaders. Naga is an optional dependency of wgpu-shaders
so these can't live in naga, and wgpu-shaders depends on naga meaning that for naga to use these types they cannot live
in wgpu-shaders.