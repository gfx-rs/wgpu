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

    encode_pass_with(device, layers, multiview_mask, None, |pass| {
        if let Some(bundle) = &bundle {
            pass.execute_bundles([bundle]);
        }
    })
}

/// Records a render pass with a depth attachment of `layers` array layers, recording its
/// commands with `record`.
fn encode_pass_with(
    device: &Device,
    layers: u32,
    multiview_mask: u32,
    occlusion_query_set: Option<&QuerySet>,
    record: impl FnOnce(&mut RenderPass<'_>),
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
        texture_binding_view_dimension: None,
    });
    let view = texture.create_view(&TextureViewDescriptor {
        dimension: Some(if layers > 1 {
            TextureViewDimension::D2Array
        } else {
            TextureViewDimension::D2
        }),
        ..Default::default()
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
            occlusion_query_set,
            multiview_mask: NonZeroU32::new(multiview_mask),
        });
        record(&mut pass);
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

fn query_set(device: &Device, ty: QueryType, count: u32) -> QuerySet {
    device.create_query_set(&QuerySetDescriptor {
        label: None,
        ty,
        count,
    })
}

#[test]
fn occlusion_queries_in_multiview_pass_use_one_slot_per_view() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    let set = query_set(&device, QueryType::Occlusion, 4);
    let encoder = encode_pass_with(&device, 2, 0b11, Some(&set), |pass| {
        pass.begin_occlusion_query(0);
        pass.end_occlusion_query();
        pass.begin_occlusion_query(2);
        pass.end_occlusion_query();
    });
    valid(&device, || encoder.finish());
}

#[test]
fn occlusion_queries_in_multiview_pass_with_overlapping_slots() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    let set = query_set(&device, QueryType::Occlusion, 4);
    let encoder = encode_pass_with(&device, 2, 0b11, Some(&set), |pass| {
        pass.begin_occlusion_query(0);
        pass.end_occlusion_query();
        pass.begin_occlusion_query(1);
        pass.end_occlusion_query();
    });
    fail(
        &device,
        || encoder.finish(),
        Some("Query 1 has already been used within the same renderpass"),
    );
}

#[test]
fn occlusion_query_in_multiview_pass_overrunning_query_set() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    let set = query_set(&device, QueryType::Occlusion, 2);
    let encoder = encode_pass_with(&device, 2, 0b11, Some(&set), |pass| {
        pass.begin_occlusion_query(1);
        pass.end_occlusion_query();
    });
    fail(
        &device,
        || encoder.finish(),
        Some("Query 1 uses 2 slots because the render pass has 2 multiview views"),
    );
}

#[test]
fn timestamp_in_multiview_pass_overrunning_query_set() {
    let (device, _queue) = multiview_device(
        Features::MULTIVIEW | Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        2,
    );
    let set = query_set(&device, QueryType::Timestamp, 2);
    let encoder = encode_pass_with(&device, 2, 0b11, None, |pass| {
        pass.write_timestamp(&set, 1);
    });
    fail(
        &device,
        || encoder.finish(),
        Some("Query 1 uses 2 slots because the render pass has 2 multiview views"),
    );
}

#[test]
fn pipeline_statistics_query_in_multiview_pass_overrunning_query_set() {
    let (device, _queue) =
        multiview_device(Features::MULTIVIEW | Features::PIPELINE_STATISTICS_QUERY, 2);
    let set = query_set(
        &device,
        QueryType::PipelineStatistics(PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS),
        2,
    );
    let encoder = encode_pass_with(&device, 2, 0b11, None, |pass| {
        pass.begin_pipeline_statistics_query(&set, 1);
        pass.end_pipeline_statistics_query();
    });
    fail(
        &device,
        || encoder.finish(),
        Some("Query 1 uses 2 slots because the render pass has 2 multiview views"),
    );
}

#[test]
fn render_pass_multi_layer_attachment_requires_multiview_mask() {
    let (device, _queue) = multiview_device(Features::MULTIVIEW, 2);
    let encoder = encode_pass(&device, 2, 0, None);
    fail(
        &device,
        || encoder.finish(),
        Some("Attachments with more than one array layer require a multiview mask"),
    );
}
