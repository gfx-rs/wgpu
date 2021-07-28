// This snapshot tests accessing various containers, dereferencing pointers.

[[block]]
struct Bar {
	matrix: mat4x4<f32>;
	arr: [[stride(8)]] array<vec2<u32>, 2>;
	data: [[stride(4)]] array<i32>;
};

[[group(0), binding(0)]]
var<storage,read_write> bar: Bar;

[[stage(vertex)]]
fn foo([[builtin(vertex_index)]] vi: u32) -> [[builtin(position)]] vec4<f32> {
    var foo: f32 = 0.0;
    // We should check that backed doesn't skip this expression
    let baz: f32 = foo;
    foo = 1.0;

    // test storage loads
	let matrix = bar.matrix;
	let arr = bar.arr;
	let index = 3u;
	let b = bar.matrix[index].x;
	let a = bar.data[arrayLength(&bar.data) - 2u];

	// test storage stores
	bar.matrix[1].z = 1.0;
	bar.matrix = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(2.0), vec4<f32>(3.0));
	bar.arr = array<vec2<u32>, 2>(vec2<u32>(0u), vec2<u32>(1u));

	// test array indexing
	var c = array<i32, 5>(a, i32(b), 3, 4, 5);
	c[vi + 1u] = 42;
	let value = c[vi];

	return matrix * vec4<f32>(vec4<i32>(value));
}
