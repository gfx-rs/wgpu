override width = 8u;
override height = 2u;
override count = 37u;

struct Inner {
    values: array<array<vec4<f32>, 3>, 5>,
    atoms: array<atomic<u32>, 19>,
    matrix: mat3x2<f32>,
    matrices: array<mat2x2<f32>, 3>,
}

struct Tagged {
    @location(0) matrix: mat2x2<f32>,
}

var<workgroup> data: array<Inner, 3>;
var<workgroup> dynamic: array<u32, count>;
var<workgroup> floats: array<f32, count>;
var<workgroup> tagged: Tagged;
var<workgroup> scalar: u32;
var<workgroup> vector: vec3<u32>;
var<workgroup> matrix: mat2x3<f32>;
var<workgroup> unused: array<u32, 1024>;

@group(0) @binding(0) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(width, height, 1)
fn main(
    @builtin(local_invocation_index) zero_flat_index: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    var ok = scalar == 0u && all(vector == vec3(0u));
    ok &= all(matrix[0] == vec3(0.0)) && all(matrix[1] == vec3(0.0));
    ok &= all(tagged.matrix[0] == vec2(0.0)) && all(tagged.matrix[1] == vec2(0.0));
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 5u; j++) {
            for (var k = 0u; k < 3u; k++) {
                ok &= all(data[i].values[j][k] == vec4(0.0));
            }
        }
        for (var j = 0u; j < 19u; j++) {
            ok &= atomicLoad(&data[i].atoms[j]) == 0u;
        }
        for (var j = 0u; j < 3u; j++) {
            ok &= all(data[i].matrix[j] == vec2(0.0));
            ok &= all(data[i].matrices[j][0] == vec2(0.0));
            ok &= all(data[i].matrices[j][1] == vec2(0.0));
        }
    }
    for (var i = 0u; i < count; i++) {
        ok &= dynamic[i] == 0u;
        ok &= floats[i] == 0.0;
    }
    workgroupBarrier();
    if zero_flat_index == 0u {
        scalar = 42u;
        vector = vec3(42u);
        matrix = mat2x3(vec3(42.0), vec3(42.0));
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < 5u; j++) {
                for (var k = 0u; k < 3u; k++) {
                    data[i].values[j][k] = vec4(42.0);
                }
            }
            for (var j = 0u; j < 19u; j++) {
                atomicStore(&data[i].atoms[j], 42u);
            }
        }
        for (var i = 0u; i < count; i++) {
            dynamic[i] = 42u;
            floats[i] = 42.0;
        }
    }
    workgroupBarrier();
    ok &= scalar == 42u && all(vector == vec3(42u));
    ok &= all(matrix[0] == vec3(42.0));
    ok &= all(data[2].values[4][2] == vec4(42.0));
    ok &= atomicLoad(&data[2].atoms[18]) == 42u;
    ok &= dynamic[count - 1u] == 42u;
    ok &= floats[count - 1u] == 42.0;
    let index = group.x * width * height + zero_flat_index;
    output[index] |= select(1u, 2u, ok);
}

@compute @workgroup_size(1)
fn single() {
    output[0] = scalar;
}
