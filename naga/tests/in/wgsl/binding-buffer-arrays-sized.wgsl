// A storage binding array whose element type has no runtime-sized
// member, with its size overridden by the binding map. The element
// struct must still be decorated `Block`.
enable wgpu_binding_array;

struct Foo { x: u32 }

@group(0) @binding(0)
var<storage, read> storage_array: binding_array<Foo>;

@fragment
fn main() -> @location(0) u32 {
    return storage_array[0].x;
}
