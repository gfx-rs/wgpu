@id(0) override has_point_light: bool = true;
@id(1200) override specular_param: f32 = 2.3f;
@id(1300) override gain: f32;
override width: f32 = 0f;
override depth: f32;
override height: f32 = (2f * depth);
override inferred_f32_: f32 = 2.718f;

var<private> gain_x_10_: f32 = (gain * 10f);
var<private> store_override: f32;

@compute @workgroup_size(1, 1, 1) 
fn main() {
    var t: f32 = (height * 5f);
    var x: bool;
    var gain_x_100_: f32;

    let a = !(has_point_light);
    x = a;
    let _e7 = gain_x_10_;
    gain_x_100_ = (_e7 * 10f);
    store_override = gain;
    return;
}
