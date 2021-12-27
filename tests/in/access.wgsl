// This snapshot tests accessing various containers, dereferencing pointers.

struct Bar {
	matrix: mat4x4<f32>;
	matrix_array: array<mat2x2<f32>, 2>;
	atom: atomic<i32>;
	arr: [[stride(8)]] array<vec2<u32>, 2>;
	data: [[stride(8)]] array<i32>;
};

[[group(0), binding(0)]]
var<storage,read_write> bar: Bar;

fn read_from_private(foo: ptr<function, f32>) -> f32 {
    return *foo;
}

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

	// test pointer types
	let data_pointer: ptr<storage, i32, read_write> = &bar.data[0];
	let foo_value = read_from_private(&foo);

	// test storage stores
	bar.matrix[1].z = 1.0;
	bar.matrix = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(1.0), vec4<f32>(2.0), vec4<f32>(3.0));
	bar.arr = array<vec2<u32>, 2>(vec2<u32>(0u), vec2<u32>(1u));
	bar.data[1] = 1;

	// test array indexing
	var c = array<i32, 5>(a, i32(b), 3, 4, 5);
	c[vi + 1u] = 42;
	let value = c[vi];

	return matrix * vec4<f32>(vec4<i32>(value));
}

[[stage(compute), workgroup_size(1)]]
fn atomics() {
	var tmp: i32;
	let value = atomicLoad(&bar.atom);
	tmp = atomicAdd(&bar.atom, 5);
	tmp = atomicSub(&bar.atom, 5);
	tmp = atomicAnd(&bar.atom, 5);
	tmp = atomicOr(&bar.atom, 5);
	tmp = atomicXor(&bar.atom, 5);
	tmp = atomicMin(&bar.atom, 5);
	tmp = atomicMax(&bar.atom, 5);
	tmp = atomicExchange(&bar.atom, 5);
	// https://github.com/gpuweb/gpuweb/issues/2021
	// tmp = atomicCompareExchangeWeak(&bar.atom, 5, 5);
	atomicStore(&bar.atom, value);
}
