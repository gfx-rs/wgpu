struct VertexOutput {
    @builtin(position) gl_Position: vec4<f32>,
}

var<private> gl_Position: vec4<f32>;

fn main_1() {
    var m: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1f, 0f, 0f), vec3<f32>(0f, 1f, 0f), vec3<f32>(0f, 0f, 1f));

    m[0i][1] = 2f;
    m[1i][0] = 3f;
    m[1i][2] = 4f;
    let _e20 = m[1];
    let _e21 = _e20.xyz;
    m[2i][2] = _e21.x;
    m[2i][1] = _e21.y;
    m[2i][0] = _e21.z;
    let _e30 = m[0];
    let _e33 = m[1];
    let _e36 = m[2];
    gl_Position = vec4<f32>(_e30.x, _e33.y, _e36.z, 1f);
    return;
}

@vertex
fn main() -> VertexOutput {
    main_1();
    let _e1 = gl_Position;
    return VertexOutput(_e1);
}
