var<private> global: vec4<f32>;
var<private> entryPointParam_fs_main: vec4<f32>;
@group(0) @binding(0) 
var texture: texture_multisampled_2d<f32>;

fn fs_main() {
    var index: i32;
    var color: vec4<f32>;

    let _e9 = global;
    index = 0i;
    loop {
        let _e12 = index;
        if (_e12 < 8i) {
        } else {
            break;
        }
        let _e14 = index;
        let _e15 = textureLoad(texture, vec2<i32>(_e9.xy), _e14);
        let _e16 = color;
        let _e18 = index;
        index = (_e18 + 1i);
        color = (_e16 + _e15);
        continue;
    }
    let _e20 = color;
    entryPointParam_fs_main = (_e20 * vec4<f32>(0.125f, 0.125f, 0.125f, 0.125f));
    return;
}

@fragment 
fn main(@builtin(position) param: vec4<f32>) -> @location(0) vec4<f32> {
    global = param;
    fs_main();
    let _e3 = entryPointParam_fs_main;
    return _e3;
}
