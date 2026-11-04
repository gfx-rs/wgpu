//! Tests of multiview validation.

use core::num::NonZeroU32;

use wgpu::*;
use wgpu_test::{fail, valid};

fn multiview_device(features: Features, max_multiview_view_count: u32) -> (Device, Queue) {
    Device::noop(&DeviceDescriptor {
        required_features: features,
        required_limits: Limits {
            max_multiview_view_count,
            ..Limits::default()
        },
        ..Default::default()
    })
}

/// Records a render pass with a depth attachment of `layers` array layers, optionally executing
/// a render bundle created with `bundle_mask`.
fn encode_pass(
    device: &Device,
    layers: u32,
    multiview_mask: u32,
    bundle_mask: Option<u32>,
) -> CommandEncoder {
    let texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(if layers > 1 {
            TextureViewDimension::D2Array
        } else {
            TextureViewDimension::D2
        }),
        ..Default::default()
    });

    let bundle = bundle_mask.map(|mask| {
        device
            .create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
                label: None,
                color_formats: &[],
                depth_stencil: Some(RenderBundleDepthStencil {
                    format: TextureFormat::Depth32Float,
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                sample_count: 1,
                multiview_mask: NonZeroU32::new(mask),
            })
            .finish(&RenderBundleDescriptor::default())
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: NonZeroU32::new(multiview_mask),
        });
        if let Some(bundle) = &bundle {
            pass.execute_bundles([bundle]);
        }
    }
    encoder
}

fn create_pipeline(device: &Device, multiview_mask: u32) -> RenderPipeline {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0); }".into(),
        ),
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: VertexState {
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        fragment: None,
        multiview_mask: NonZeroU32::new(multiview_mask),
        cache: None,
    })
}

#[test]
fn render_pass_multiview_mask_requires_multiview_feature() {
    let (device, _queue) = multiview_device(Features::empty(), 0);
    let encoder = encode_pass(&device, 1, 0b1, None);
    fail(&device, || encoder.finish(), Some("MULTIVIEW"));
}

#[test]
fn render_pass_multiview_mask_with_multiview_feature() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 1);
    let encoder = encode_pass(&device, 1, 0b1, None);
    valid(&device, || encoder.finish());
}

#[test]
fn render_pass_multiview_mask_covering_32_layers() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 32);
    let encoder = encode_pass(&device, 32, u32::MAX, None);
    valid(&device, || encoder.finish());
}

#[test]
fn render_pass_multiview_mask_not_covering_all_layers_requires_selective_multiview() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 4);
    let encoder = encode_pass(&device, 4, 0b11, None);
    fail(&device, || encoder.finish(), Some("SELECTIVE_MULTIVIEW"));
}

#[test]
fn render_pipeline_multiview_mask_of_all_bits() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 32);
    valid(&device, || create_pipeline(&device, u32::MAX));
}

#[test]
fn render_pipeline_multiview_mask_skipping_layer_0_requires_selective_multiview() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    fail(
        &device,
        || create_pipeline(&device, 0b10),
        Some("SELECTIVE_MULTIVIEW"),
    );
}

#[test]
fn render_bundle_multiview_mask_matches_render_pass() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    let encoder = encode_pass(&device, 2, 0b11, Some(0b11));
    valid(&device, || encoder.finish());
}

#[test]
fn render_bundle_multiview_mask_mismatching_render_pass() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    let encoder = encode_pass(&device, 2, 0b11, Some(0b01));
    fail(&device, || encoder.finish(), Some("multiview"));
}
