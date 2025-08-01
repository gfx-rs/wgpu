@compute @workgroup_size(1)
fn f() {
    let b = array<vec3f, 2>();
    var poly = vec4f(0);
    var k = 0;
    var j = 0;

    poly.x += b[j].y * b[k].z;
}
