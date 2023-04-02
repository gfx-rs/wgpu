#[path = "../framework.rs"]
mod framework;

use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, f32::consts, mem};
use wgpu::util::DeviceExt;

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const MIP_LEVEL_COUNT: u32 = 10;
const MIP_PASS_COUNT: u32 = MIP_LEVEL_COUNT - 1;

fn create_texels(size: usize, cx: f32, cy: f32) -> Vec<u8> {
    use std::iter;

    (0..size * size)
        .flat_map(|id| {
            // get high five for recognizing this ;)
            let mut x = 4.0 * (id % size) as f32 / (size - 1) as f32 - 2.0;
            let mut y = 2.0 * (id / size) as f32 / (size - 1) as f32 - 1.0;
            let mut count = 0;
            while count < 0xFF && x * x + y * y < 4.0 {
                let old_x = x;
                x = x * x - y * y + cx;
                y = 2.0 * old_x * y + cy;
                count += 1;
            }
            iter::once(0xFF - (count * 2) as u8)
                .chain(iter::once(0xFF - (count * 5) as u8))
                .chain(iter::once(0xFF - (count * 13) as u8))
                .chain(iter::once(std::u8::MAX))
        })
        .collect()
}

struct QuerySets {
    timestamp: wgpu::QuerySet,
    timestamp_period: f32,
    pipeline_statistics: wgpu::QuerySet,
    data_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimestampData {
    start: u64,
    end: u64,
}

type TimestampQueries = [TimestampData; MIP_PASS_COUNT as usize];
type PipelineStatisticsQueries = [u64; MIP_PASS_COUNT as usize];

fn pipeline_statistics_offset() -> wgpu::BufferAddress {
    (mem::size_of::<TimestampQueries>() as wgpu::BufferAddress)
        .max(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT)
}

struct Example {
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    draw_pipeline: wgpu::RenderPipeline,
}

impl Example {
    fn generate_matrix(aspect_ratio: f32) -> glam::Mat4 {
        let projection = glam::Mat4::perspective_rh(consts::FRAC_PI_4, aspect_ratio, 1.0, 1000.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0f32, 0.0, 10.0),
            glam::Vec3::new(0f32, 50.0, 0.0),
            glam::Vec3::Z,
        );
        projection * view
    }

    fn generate_mipmaps(
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        query_sets: &Option<QuerySets>,
        mip_count: u32,
    ) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("blit.wgsl"))),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(TEXTURE_FORMAT.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let views = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("mip"),
                    format: None,
                    dimension: None,
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: None,
                })
            })
            .collect::<Vec<_>>();

        for target_mip in 1..mip_count as usize {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[target_mip - 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: None,
            });

            let pipeline_query_index_base = target_mip as u32 - 1;
            let timestamp_query_index_base = (target_mip as u32 - 1) * 2;

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &views[target_mip],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: &[],
            });
            if let Some(ref query_sets) = query_sets {
                rpass.write_timestamp(&query_sets.timestamp, timestamp_query_index_base);
                rpass.begin_pipeline_statistics_query(
                    &query_sets.pipeline_statistics,
                    pipeline_query_index_base,
                );
            }
            rpass.set_pipeline(&pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
            if let Some(ref query_sets) = query_sets {
                rpass.write_timestamp(&query_sets.timestamp, timestamp_query_index_base + 1);
                rpass.end_pipeline_statistics_query();
            }
        }

        if let Some(ref query_sets) = query_sets {
            let timestamp_query_count = MIP_PASS_COUNT * 2;
            encoder.resolve_query_set(
                &query_sets.timestamp,
                0..timestamp_query_count,
                &query_sets.data_buffer,
                0,
            );
            encoder.resolve_query_set(
                &query_sets.pipeline_statistics,
                0..MIP_PASS_COUNT,
                &query_sets.data_buffer,
                pipeline_statistics_offset(),
            );
        }
    }
}

impl framework::Example for Example {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::PIPELINE_STATISTICS_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
    }

    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut init_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // Create the texture
        let size = 1 << MIP_PASS_COUNT;
        let texels = create_texels(size as usize, -0.8, 0.156);
        let texture_extent = wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: MIP_LEVEL_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST,
            label: None,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        //Note: we could use queue.write_texture instead, and this is what other
        // examples do, but here we want to show another way to do this.
        let temp_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Temporary Buffer"),
            contents: texels.as_slice(),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        init_encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &temp_buf,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * size),
                    rows_per_image: None,
                },
            },
            texture.as_image_copy(),
            texture_extent,
        );

        // Create other resources
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(mx_ref),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create the render pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw.wgsl"))),
        });

        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("draw"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create bind group
        let bind_group_layout = draw_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });

        // If both kinds of query are supported, use queries
        let query_sets = if device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::PIPELINE_STATISTICS_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ) {
            // For N total mips, it takes N - 1 passes to generate them, and we're measuring those.
            let mip_passes = MIP_LEVEL_COUNT - 1;

            // Create the timestamp query set. We need twice as many queries as we have passes,
            // as we need a query at the beginning and at the end of the operation.
            let timestamp = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                count: mip_passes * 2,
                ty: wgpu::QueryType::Timestamp,
            });
            // Timestamp queries use an device-specific timestamp unit. We need to figure out how many
            // nanoseconds go by for the timestamp to be incremented by one. The period is this value.
            let timestamp_period = queue.get_timestamp_period();

            // We only need one pipeline statistics query per pass.
            let pipeline_statistics = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                count: mip_passes,
                ty: wgpu::QueryType::PipelineStatistics(
                    wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
                ),
            });

            // This databuffer has to store all of the query results, 2 * passes timestamp queries
            // and 1 * passes statistics queries. Each query returns a u64 value.
            let data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query buffer"),
                size: pipeline_statistics_offset()
                    + mem::size_of::<PipelineStatisticsQueries>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            Some(QuerySets {
                timestamp,
                timestamp_period,
                pipeline_statistics,
                data_buffer,
            })
        } else {
            None
        };

        Self::generate_mipmaps(
            &mut init_encoder,
            device,
            &texture,
            &query_sets,
            MIP_LEVEL_COUNT,
        );

        queue.submit(Some(init_encoder.finish()));
        if let Some(ref query_sets) = query_sets {
            // We can ignore the callback as we're about to wait for the device.
            query_sets
                .data_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            // Wait for device to be done rendering mipmaps
            device.poll(wgpu::Maintain::Wait);
            // This is guaranteed to be ready.
            let timestamp_view = query_sets
                .data_buffer
                .slice(..mem::size_of::<TimestampQueries>() as wgpu::BufferAddress)
                .get_mapped_range();
            let pipeline_stats_view = query_sets
                .data_buffer
                .slice(pipeline_statistics_offset()..)
                .get_mapped_range();
            // Convert the raw data into a useful structure
            let timestamp_data: &TimestampQueries = bytemuck::from_bytes(&timestamp_view);
            let pipeline_stats_data: &PipelineStatisticsQueries =
                bytemuck::from_bytes(&pipeline_stats_view);
            // Iterate over the data
            for (idx, (timestamp, pipeline)) in timestamp_data
                .iter()
                .zip(pipeline_stats_data.iter())
                .enumerate()
            {
                // Figure out the timestamp differences and multiply by the period to get nanoseconds
                let nanoseconds =
                    (timestamp.end - timestamp.start) as f32 * query_sets.timestamp_period;
                // Nanoseconds is a bit small, so lets use microseconds.
                let microseconds = nanoseconds / 1000.0;
                // Print the data!
                println!(
                    "Generating mip level {} took {:.3} μs and called the fragment shader {} times",
                    idx + 1,
                    microseconds,
                    pipeline
                );
            }
        }

        Example {
            bind_group,
            uniform_buf,
            draw_pipeline,
        }
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let mx_total = Self::generate_matrix(config.width as f32 / config.height as f32);
        let mx_ref: &[f32; 16] = mx_total.as_ref();
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(mx_ref));
    }

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let clear_color = wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            };
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: &[],
            });
            rpass.set_pipeline(&self.draw_pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);
            rpass.draw(0..4, 0..1);
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("mipmap");
}

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn mipmap() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/mipmap/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::default(),
        base_test_parameters: framework::test_common::TestParameters::default()
            .backend_failure(wgpu::Backends::GL),
        tolerance: 50,
        max_outliers: 5000, // Mipmap sampling is highly variant between impls. This is currently bounded by lavapipe
    });
}
