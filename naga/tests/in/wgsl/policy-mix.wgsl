// Tests that the index, buffer, and texture bounds checks policies are
// implemented separately.

// Storage and Uniform storage classes
struct InStorage {
  a: array<vec4<f32>, 10>
}
@group(0) @binding(0) var<storage> in_storage: InStorage;

struct InUniform {
  a: array<vec4<f32>, 20>
}
@group(0) @binding(1) var<uniform> in_uniform: InUniform;

// Textures automatically land in the `handle` storage class.
@group(0) @binding(2) var image_2d_array: texture_2d_array<f32>;

// None of the above.
var<workgroup> in_workgroup: array<f32, 30>;
var<private> in_private: array<f32, 40>;

fn mock_function(c: vec2<i32>, i: i32, l: i32) -> vec4<f32> {
  var in_function: array<vec4<f32>, 2> =
    array<vec4<f32>, 2>(vec4<f32>(0.707, 0.0, 0.0, 1.0),
                        vec4<f32>(0.0, 0.707, 0.0, 1.0));

  return (in_storage.a[i] +
          in_uniform.a[i] +
          textureLoad(image_2d_array, c, i, l) +
          in_workgroup[i] +
          in_private[i] +
          in_function[i]);
}
