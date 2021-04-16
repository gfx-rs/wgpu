// This snapshot tests accessing various containers, dereferencing pointers.

[[stage(vertex)]]
fn foo([[builtin(vertex_index)]] vi: u32) -> [[builtin(position)]] vec4<f32> {
	let array = array<i32, 5>(1, 2, 3, 4, 5);
	let value = array[vi];
	return vec4<f32>(vec4<i32>(value));
}
