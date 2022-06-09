use core::default::Default;
use naga::ShaderStage;
use std::borrow::Cow;
use wgpu::{ShaderModuleDescriptor, ShaderSource};

use crate::common::{initialize_test, TestParameters};

#[test]
fn dont_panic() {
    initialize_test(TestParameters::default().test_features_limits(), |ctx| {
        let glsl = include_str!("invalid_shader.glsl");
        let shader = ctx.device.create_shader_module(&ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Glsl {
                shader: Cow::Borrowed(glsl),
                stage: ShaderStage::Vertex,
                defines: Default::default(),
            },
        });
        assert!(shader.is_err());
    })
}
