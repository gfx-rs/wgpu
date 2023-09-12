fn main_1() {
    var splat: mat2x2<f32>;
    var normal: mat2x2<f32>;
    var from_matrix: mat2x4<f32>;
    var a: mat2x2<f32>;
    var b: mat2x2<f32>;
    var c: mat3x3<f32>;
    var d: mat3x3<f32>;
    var e: mat4x4<f32>;

    let _e1 = f32(1);
    splat = mat2x2<f32>(vec2<f32>(_e1, 0.0), vec2<f32>(0.0, _e1));
    let _e9 = vec2(f32(1));
    let _e12 = vec2(f32(2));
    normal = mat2x2<f32>(vec2<f32>(_e9.x, _e9.y), vec2<f32>(_e12.x, _e12.y));
    let _e26 = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    from_matrix = mat2x4<f32>(vec4<f32>(_e26[0].x, _e26[0].y, _e26[0].z, 0.0), vec4<f32>(_e26[1].x, _e26[1].y, _e26[1].z, 0.0));
    a = mat2x2<f32>(vec2<f32>(f32(1), f32(2)), vec2<f32>(f32(3), f32(4)));
    let _e58 = vec2<f32>(f32(2), f32(3));
    b = mat2x2<f32>(vec2<f32>(f32(1), _e58.x), vec2<f32>(_e58.y, f32(4)));
    let _e73 = vec3(f32(1));
    let _e76 = vec3(f32(1));
    c = mat3x3<f32>(vec3<f32>(f32(1), f32(2), f32(3)), vec3<f32>(_e73.x, _e73.y, _e73.z), vec3<f32>(_e76.x, _e76.y, _e76.z));
    let _e93 = vec2(f32(2));
    let _e97 = vec3(f32(1));
    let _e100 = vec3(f32(1));
    d = mat3x3<f32>(vec3<f32>(_e93.x, _e93.y, f32(1)), vec3<f32>(_e97.x, _e97.y, _e97.z), vec3<f32>(_e100.x, _e100.y, _e100.z));
    let _e117 = vec2(f32(2));
    let _e120 = vec4(f32(1));
    let _e123 = vec2(f32(2));
    let _e126 = vec4(f32(1));
    let _e129 = vec4(f32(1));
    e = mat4x4<f32>(vec4<f32>(_e117.x, _e117.y, _e120.x, _e120.y), vec4<f32>(_e120.z, _e120.w, _e123.x, _e123.y), vec4<f32>(_e126.x, _e126.y, _e126.z, _e126.w), vec4<f32>(_e129.x, _e129.y, _e129.z, _e129.w));
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}
