@fragment
fn fs_main() -> @location(0) vec4f {
    let my_array = array(
        vec2f(0.0, 0.0),
        vec2f(0.0, 0.0),
    );
    var index_0 = 0;

    let val_0 = my_array[index_0];
    let val_1 = my_array[index_0];

    return (val_0 * val_1).xxyy;
}
