struct QEFResult {
        a: f32,
        b: vec3<f32>,
}

fn foobar(normals: array<vec3f, 12>, count: u32) -> QEFResult {
        for (var i = 0u; i < count; i++) {
                var n0 = normals[i];
        }

        for (var j = 0u; j < count; j++) {
                var n1 = normals[j];
        }

        return QEFResult(0.0, vec3(0.0));
}

@fragment
fn main() {
    var arr: array<vec3f, 12>;
    foobar(arr, 1);
}
