/*!
Test the GLSL backend's reflection info.
*/

#![cfg(all(feature = "wgsl-in", glsl_out))]

use naga::back::glsl::{self, BufferKind, BufferMapping, GlobalResource, ReflectionInfo};
use naga::{ImageClass, ImageDimension, ResourceBinding};

const SOURCE: &str = "
@group(3) @binding(0) var unused_tex: texture_2d<f32>;
@group(0) @binding(0) var<uniform> u: vec4<f32>;
@group(0) @binding(1) var<storage, read_write> s: array<vec4<f32>>;
@group(1) @binding(0) var t: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var st: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main() {
    let c = textureSampleLevel(t, samp, vec2(0.5), 0.0);
    s[0] = u + c;
    textureStore(st, vec2(0), c);
}
";

fn reflect(module: &naga::Module) -> ReflectionInfo {
    naga::valid::Validator::new(Default::default(), Default::default())
        .validate(module)
        .unwrap();
    let pipeline_options = glsl::PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".to_string(),
        multiview: None,
    };
    glsl::Writer::new(
        String::new(),
        &glsl::Options::default(),
        &pipeline_options,
        Default::default(),
    )
    .write(module)
    .unwrap()
}

fn resource(group: u32, binding: u32, name: &str) -> GlobalResource {
    GlobalResource {
        binding: Some(ResourceBinding { group, binding }),
        name: Some(name.to_string()),
    }
}

#[test]
fn resources_by_binding() {
    let module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let reflection = reflect(&module);

    let mut uniforms = reflection.uniforms.clone();
    uniforms.sort_by_key(|b| b.global.binding);
    let globals: Vec<_> = uniforms.iter().map(|b| (&b.global, b.kind)).collect();
    assert_eq!(
        globals,
        [
            (&resource(0, 0, "u"), BufferKind::Uniform),
            (&resource(0, 1, "s"), BufferKind::Storage),
        ]
    );

    assert_eq!(reflection.texture_mapping.len(), 2);
    let sampled = reflection
        .texture_mapping
        .values()
        .find(|m| m.texture == resource(1, 0, "t"))
        .unwrap();
    assert_eq!(sampled.sampler, Some(resource(1, 1, "samp")));
    assert_eq!(sampled.dim, ImageDimension::D2);
    assert!(!sampled.arrayed);
    assert!(matches!(sampled.class, ImageClass::Sampled { multi: false, .. }));

    let storage = reflection
        .texture_mapping
        .values()
        .find(|m| m.texture == resource(1, 2, "st"))
        .unwrap();
    assert_eq!(storage.sampler, None);
    assert!(matches!(storage.class, ImageClass::Storage { .. }));
}

#[test]
fn unchanged_by_compaction() {
    let mut module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let before = reflect(&module);

    let global_count = module.global_variables.len();
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    assert!(module.global_variables.len() < global_count);
    let after = reflect(&module);

    let sorted = |buffers: &[BufferMapping]| {
        let mut buffers = buffers.to_vec();
        buffers.sort_by_key(|b| b.global.binding);
        buffers
    };
    assert_eq!(sorted(&before.uniforms), sorted(&after.uniforms));
    assert_eq!(before.texture_mapping, after.texture_mapping);
}
