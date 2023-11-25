// Create several type definitions to test `align` and `size` layout.

struct NoPadding {
	@location(0)
	v3: vec3f, // align 16, size 12; no start padding needed
	@location(1)
	f3: f32, // align 4, size 4; no start padding needed
}
@fragment
fn no_padding_frag(input: NoPadding) -> @location(0) vec4f {
	_ = input;
	return vec4f(0.0);
}
@vertex
fn no_padding_vert(input: NoPadding) -> @builtin(position) vec4f {
	_ = input;
	return vec4f(0.0);
}
@group(0) @binding(0) var<uniform> no_padding_uniform: NoPadding;
@group(0) @binding(1) var<storage, read_write> no_padding_storage: NoPadding;
@compute @workgroup_size(16,1,1)
fn no_padding_comp() {
	var x: NoPadding;
	x = no_padding_uniform;
	x = no_padding_storage;
}

struct NeedsPadding {
	@location(0) f3_forces_padding: f32, // align 4, size 4; no start padding needed
	@location(1) v3_needs_padding: vec3f, // align 16, size 12; needs 12 bytes of padding
	@location(2) f3: f32, // align 4, size 4; no start padding needed
}
@fragment
fn needs_padding_frag(input: NeedsPadding) -> @location(0) vec4f {
	_ = input;
	return vec4f(0.0);
}
@vertex
fn needs_padding_vert(input: NeedsPadding) -> @builtin(position) vec4f {
	_ = input;
	return vec4f(0.0);
}
@group(0) @binding(2) var<uniform> needs_padding_uniform: NeedsPadding;
@group(0) @binding(3) var<storage, read_write> needs_padding_storage: NeedsPadding;
@compute @workgroup_size(16,1,1)
fn needs_padding_comp() {
	var x: NeedsPadding;
	x = needs_padding_uniform;
	x = needs_padding_storage;
}
