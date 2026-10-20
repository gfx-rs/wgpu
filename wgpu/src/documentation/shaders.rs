/*!
## Shader Support

`wgpu` can consume shaders in [WGSL](https://gpuweb.github.io/gpuweb/wgsl/), SPIR-V, and GLSL.
Both [HLSL](https://github.com/Microsoft/DirectXShaderCompiler) and [GLSL](https://github.com/KhronosGroup/glslang)
have compilers to target SPIR-V. All of these shader languages can be used with any backend as we handle all of the conversions. Additionally, support for these shader inputs is not going away.

While WebGPU does not support any shading language other than WGSL, we will automatically convert your
non-WGSL shaders if you're running on WebGPU.

WGSL is always supported by default, but GLSL and SPIR-V need features enabled to compile in support.

To enable WGSL shaders, enable the `wgsl` feature of `wgpu` (enabled by default).
To enable SPIR-V shaders, enable the `spirv` feature of `wgpu`.
To enable GLSL shaders, enable the `glsl` feature of `wgpu`.
*/
