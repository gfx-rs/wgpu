// This snapshot tests accessing various containers, dereferencing pointers.

struct AlignedWrapper {
	@align(8) value: i32
}

struct Bar {
	_matrix: mat4x3<f32>,
	matrix_array: array<mat2x2<f32>, 2>,
	atom: atomic<i32>,
	arr: array<vec2<u32>, 2>,
	data: array<AlignedWrapper>,
}

@group(0) @binding(0)
var<storage,read_write> bar: Bar;

struct Baz {
	m: mat3x2<f32>,
}

@group(0) @binding(1)
var<uniform> baz: Baz;

@group(0) @binding(2)
var<storage,read_write> qux: vec2<i32>;

fn test_matrix_within_struct_accesses() {
	var idx = 1;

    idx--;

	// loads
    _ = baz.m;
    _ = baz.m[0];
    _ = baz.m[idx];
    _ = baz.m[0][1];
    _ = baz.m[0][idx];
    _ = baz.m[idx][1];
    _ = baz.m[idx][idx];

    var t = Baz(mat3x2<f32>(vec2<f32>(1.0), vec2<f32>(2.0), vec2<f32>(3.0)));

    idx++;

	// stores
    t.m = mat3x2<f32>(vec2<f32>(6.0), vec2<f32>(5.0), vec2<f32>(4.0));
    t.m[0] = vec2<f32>(9.0);
    t.m[idx] = vec2<f32>(90.0);
    t.m[0][1] = 10.0;
    t.m[0][idx] = 20.0;
    t.m[idx][1] = 30.0;
    t.m[idx][idx] = 40.0;
}

fn read_from_private(foo: ptr<function, f32>) -> f32 {
    return *foo;
}

fn test_arr_as_arg(a: array<array<f32, 10>, 5>) -> f32 {
    return a[4][9];
}

@vertex
fn foo_vert(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var foo: f32 = 0.0;
    // We should check that backed doesn't skip this expression
    let baz: f32 = foo;
    foo = 1.0;

	test_matrix_within_struct_accesses();

    // test storage loads
	let _matrix = bar._matrix;
	let arr = bar.arr;
	let index = 3u;
	let b = bar._matrix[index].x;
	let a = bar.data[arrayLength(&bar.data) - 2u].value;
	let c = qux;

	// test pointer types
	let data_pointer: ptr<storage, i32, read_write> = &bar.data[0].value;
	let foo_value = read_from_private(&foo);

	// test array indexing
	var c = array<i32, 5>(a, i32(b), 3, 4, 5);
	c[vi + 1u] = 42;
	let value = c[vi];

	_ = test_arr_as_arg(array<array<f32, 10>, 5>());

	return vec4<f32>(_matrix * vec4<f32>(vec4<i32>(value)), 2.0);
}

@fragment
fn foo_frag() -> @location(0) vec4<f32> {
	// test storage stores
	bar._matrix[1].z = 1.0;
	bar._matrix = mat4x3<f32>(vec3<f32>(0.0), vec3<f32>(1.0), vec3<f32>(2.0), vec3<f32>(3.0));
	bar.arr = array<vec2<u32>, 2>(vec2<u32>(0u), vec2<u32>(1u));
	bar.data[1].value = 1;
	qux = vec2<i32>();

	return vec4<f32>(0.0);
}

@compute @workgroup_size(1)
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

var<workgroup> val: u32;

fn assign_through_ptr_fn(p: ptr<workgroup, u32>) {
    *p = 42u;
}

@compute @workgroup_size(1)
fn assign_through_ptr() {
    assign_through_ptr_fn(&val);
}
