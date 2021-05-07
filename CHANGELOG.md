# Change Log

### v0.8.1 (2021-05-07)
- Fix buffer initialization with unaligned data sizes

### v0.8 (2021-04-29)
- See https://github.com/gfx-rs/wgpu/blob/v0.8/CHANGELOG.md#v08-2021-04-29
- Naga is the default shader conversion path on Metal, Vulkan, and OpenGL
- SPIRV-Cross is optionally enabled with "cross" feature
- All of the examples (except "texture-array") now use WGSL

### v0.7 (2021-01-31)
- See https://github.com/gfx-rs/wgpu/blob/v0.7/CHANGELOG.md#v07-2020-08-30
- Features:
	- (beta) WGSL support
	- better error messages
- API changes:
	- new `ShaderModuleDescriptor`
	- new `RenderEncoder`

### v0.6.2 (2020-11-24)
- don't panic in the staging belt if the channel is dropped
