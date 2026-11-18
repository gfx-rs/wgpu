struct Cell {
    @align(16) @size(32) matrix: mat2x2<f32>,
    values: array<vec4<f32>, 2>,
}
struct Buffer { cells: array<Cell, 2>, }
struct Scratch { cells: array<Cell, 2>, @align(32) tail: vec4<f32>, }
@group(0) @binding(0) var<uniform> input: Buffer;
@group(0) @binding(1) var<storage, read_write> output: Buffer;
var<workgroup> work: Buffer;
var<workgroup> scratch: Scratch;
var<workgroup> counter: atomic<u32>;
fn identity(value: Buffer) -> Buffer { return value; }
@compute @workgroup_size(1)
fn main(@builtin(local_invocation_index) index: u32) {
    work = identity(input);
    scratch.cells = work.cells;
    workgroupBarrier();
    let cells = workgroupUniformLoad(&scratch.cells);
    work.cells = cells;
    output = identity(work);
    output.cells[index % 2u] = scratch.cells[index % 2u];
    atomicStore(&counter, 1u);
    output.cells[0].values[0].x = f32(atomicLoad(&counter));
}
