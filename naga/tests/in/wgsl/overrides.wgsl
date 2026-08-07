@id(0)    override has_point_light: bool = true;  // Algorithmic control
@id(1200) override specular_param: f32 = 2.3;     // Numeric control
@id(1300) override gain: f32;                     // Must be overridden
          override width: f32 = 0.0;              // Specified at the API level using
                                                  // the name "width".
          override depth: f32;                    // Specified at the API level using
                                                  // the name "depth".
                                                  // Must be overridden.
          override height = 2 * depth;            // The default value
                                                  // (if not set at the API level),
                                                  // depends on another
                                                  // overridable constant.

override inferred_f32 = 2.718;

override auto_conversion: u32 = 0;

var<private> gain_x_10: f32 = gain * 10.;
var<private> store_override: f32;

// Composites with an override-expression component cannot be evaluated at
// constant evaluation time, but become constants during override processing,
// so they must still be flattened correctly then.
var<private> override_compose: vec4<f32> = vec4<f32>(vec2<f32>(gain, 1.0), vec2<f32>(2.0, 3.0));
var<private> override_compose_zero_value: vec4<f32> = vec4<f32>(vec2<f32>(), gain, 1.0);

@compute @workgroup_size(1)
fn main() {
    var t = height * 5;
    let a = !has_point_light;
    var x = a;

    var gain_x_100 = gain_x_10 * 10.;

    store_override = gain;

    _ = override_compose;
    _ = override_compose_zero_value;

    _ = specular_param;
    _ = width;
    _ = inferred_f32;
    _ = auto_conversion;
}
