struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) texcoord: vec2<f32>,
}

struct VertexInput {
  @location(0) position: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) texcoord: vec2<f32>,
}

@group(0) @binding(0) var<uniform> mvp_matrix: mat4x4<f32>;

@vertex
fn render_vertex(
  v_in: VertexInput,
  @builtin(vertex_index) v_existing_id: u32,
) -> VertexOutput
{
  var v_out: VertexOutput;
  v_out.position = v_in.position * mvp_matrix;
  v_out.color = do_lighting(v_in.position,
                            v_in.normal);
  v_out.texcoord = v_in.texcoord;
  return v_out;
}

fn do_lighting(position: vec4<f32>, normal: vec3<f32>) -> vec4<f32> {
  // blah blah blah
  return vec4<f32>(0);
}
