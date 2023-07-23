use std::{borrow::Cow, num::NonZeroU32};

use crate::common::{initialize_test, TestParameters};
use wasm_bindgen_test::*;

#[test]
#[wasm_bindgen_test]
fn partially_bounded_array() {
    initialize_test(
        TestParameters::default()
            .features(
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                    | wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
                    | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
            )
            .limits(wgpu::Limits {
                ..wgpu::Limits::downlevel_defaults()
            })
            .backend_failure(
                wgpu::Backends::GL
                    | wgpu::Backends::DX11
                    | wgpu::Backends::METAL
                    | wgpu::Backends::DX12
                    | wgpu::Backends::BROWSER_WEBGPU,
            ),
        |ctx| {
            let device = &ctx.device;

            let texture_extent = wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            };
            let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: texture_extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });

            let texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let size = std::mem::size_of::<f32>() as u64 * 4_u64;
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bind group layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },

                        count: NonZeroU32::new(4),
                    }],
                });

            let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("main"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &cs_module,
                    entry_point: "main",
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&[&texture_view]),
                }],
                layout: &bind_group_layout,
                label: Some("bind group"),
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &storage_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &staging_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: None,
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            ctx.queue.submit(Some(encoder.finish()));

            // wait for gpu
            let buffer_slice = staging_buffer.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            device.poll(wgpu::Maintain::Wait);

            if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
                let data = buffer_slice.get_mapped_range();
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                assert!(result.iter().eq(&[4.0, 3.0, 2.0, 1.0]));
                // dropped before we unmap the buffer.
                drop(data);
                staging_buffer.unmap();
            } else {
                panic!("failed!")
            }
        },
    )
}
