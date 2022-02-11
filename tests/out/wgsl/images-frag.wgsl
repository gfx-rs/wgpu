@group(0) @binding(0) 
var img1D: texture_storage_1d<rgba8unorm,read_write>;
@group(0) @binding(1) 
var img2D: texture_storage_2d<rgba8unorm,read_write>;
@group(0) @binding(2) 
var img3D: texture_storage_3d<rgba8unorm,read_write>;
@group(0) @binding(4) 
var img1DArray: texture_storage_1d_array<rgba8unorm,read_write>;
@group(0) @binding(5) 
var img2DArray: texture_storage_2d_array<rgba8unorm,read_write>;

fn testImg1D(coord: i32) {
    var coord_1: i32;
    var size: i32;
    var c: vec4<f32>;

    coord_1 = coord;
    let _e7 = textureDimensions(img1D);
    size = _e7;
    let _e10 = coord_1;
    let _e11 = textureLoad(img1D, _e10);
    c = _e11;
    let _e17 = coord_1;
    textureStore(img1D, _e17, vec4<f32>(f32(2)));
    return;
}

fn testImg1DArray(coord_2: vec2<i32>) {
    var coord_3: vec2<i32>;
    var size_1: vec2<f32>;
    var c_1: vec4<f32>;

    coord_3 = coord_2;
    let _e7 = textureDimensions(img1DArray);
    let _e8 = textureNumLayers(img1DArray);
    size_1 = vec2<f32>(vec2<i32>(_e7, _e8));
    let _e13 = coord_3;
    let _e16 = textureLoad(img1DArray, _e13.x, _e13.y);
    c_1 = _e16;
    let _e22 = coord_3;
    textureStore(img1DArray, _e22.x, _e22.y, vec4<f32>(f32(2)));
    return;
}

fn testImg2D(coord_4: vec2<i32>) {
    var coord_5: vec2<i32>;
    var size_2: vec2<f32>;
    var c_2: vec4<f32>;

    coord_5 = coord_4;
    let _e7 = textureDimensions(img2D);
    size_2 = vec2<f32>(_e7);
    let _e11 = coord_5;
    let _e12 = textureLoad(img2D, _e11);
    c_2 = _e12;
    let _e18 = coord_5;
    textureStore(img2D, _e18, vec4<f32>(f32(2)));
    return;
}

fn testImg2DArray(coord_6: vec3<i32>) {
    var coord_7: vec3<i32>;
    var size_3: vec3<f32>;
    var c_3: vec4<f32>;

    coord_7 = coord_6;
    let _e7 = textureDimensions(img2DArray);
    let _e10 = textureNumLayers(img2DArray);
    size_3 = vec3<f32>(vec3<i32>(_e7.x, _e7.y, _e10));
    let _e15 = coord_7;
    let _e18 = textureLoad(img2DArray, _e15.xy, _e15.z);
    c_3 = _e18;
    let _e24 = coord_7;
    textureStore(img2DArray, _e24.xy, _e24.z, vec4<f32>(f32(2)));
    return;
}

fn testImg3D(coord_8: vec3<i32>) {
    var coord_9: vec3<i32>;
    var size_4: vec3<f32>;
    var c_4: vec4<f32>;

    coord_9 = coord_8;
    let _e7 = textureDimensions(img3D);
    size_4 = vec3<f32>(_e7);
    let _e11 = coord_9;
    let _e12 = textureLoad(img3D, _e11);
    c_4 = _e12;
    let _e18 = coord_9;
    textureStore(img3D, _e18, vec4<f32>(f32(2)));
    return;
}

fn main_1() {
    return;
}

@stage(fragment) 
fn main() {
    main_1();
    return;
}
