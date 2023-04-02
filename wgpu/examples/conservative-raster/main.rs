#[path = "../framework.rs"]
mod framework;

use std::borrow::Cow;

const RENDER_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

struct Example {
    low_res_target: wgpu::TextureView,
    bind_group_upscale: wgpu::BindGroup,

    pipeline_triangle_conservative: wgpu::RenderPipeline,
    pipeline_triangle_regular: wgpu::RenderPipeline,
    pipeline_upscale: wgpu::RenderPipeline,
    pipeline_lines: Option<wgpu::RenderPipeline>,
    bind_group_layout_upscale: wgpu::BindGroupLayout,
}

impl Example {
    fn create_low_res_target(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        bind_group_layout_upscale: &wgpu::BindGroupLayout,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        let texture_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Low Resolution Target"),
                size: wgpu::Extent3d {
                    width: (config.width / 16).max(1),
                    height: (config.height / 16).max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: RENDER_TARGET_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nearest Neighbor Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("upscale bind group"),
            layout: bind_group_layout_upscale,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        (texture_view, bind_group)
    }
}

impl framework::Example for Example {
    fn required_features() -> wgpu::Features {
        wgpu::Features::CONSERVATIVE_RASTERIZATION
    }
    fn optional_features() -> wgpu::Features {
        wgpu::Features::POLYGON_MODE_LINE
    }
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let pipeline_layout_empty =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let shader_triangle_and_lines = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "triangle_and_lines.wgsl"
            ))),
        });

        let pipeline_triangle_conservative =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Conservative Rasterization"),
                layout: Some(&pipeline_layout_empty),
                vertex: wgpu::VertexState {
                    module: &shader_triangle_and_lines,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_triangle_and_lines,
                    entry_point: "fs_main_red",
                    targets: &[Some(RENDER_TARGET_FORMAT.into())],
                }),
                primitive: wgpu::PrimitiveState {
                    conservative: true,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let pipeline_triangle_regular =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Regular Rasterization"),
                layout: Some(&pipeline_layout_empty),
                vertex: wgpu::VertexState {
                    module: &shader_triangle_and_lines,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_triangle_and_lines,
                    entry_point: "fs_main_blue",
                    targets: &[Some(RENDER_TARGET_FORMAT.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        let pipeline_lines = if device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
        {
            Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Lines"),
                    layout: Some(&pipeline_layout_empty),
                    vertex: wgpu::VertexState {
                        module: &shader_triangle_and_lines,
                        entry_point: "vs_main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader_triangle_and_lines,
                        entry_point: "fs_main_white",
                        targets: &[Some(config.view_formats[0].into())],
                    }),
                    primitive: wgpu::PrimitiveState {
                        polygon_mode: wgpu::PolygonMode::Line,
                        topology: wgpu::PrimitiveTopology::LineStrip,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                }),
            )
        } else {
            None
        };

        let (pipeline_upscale, bind_group_layout_upscale) = {
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("upscale bindgroup"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("upscale.wgsl"))),
            });
            (
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Upscale"),
                    layout: Some(&pipeline_layout),
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
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                }),
                bind_group_layout,
            )
        };

        let (low_res_target, bind_group_upscale) =
            Self::create_low_res_target(config, device, &bind_group_layout_upscale);

        Self {
            low_res_target,
            bind_group_upscale,

            pipeline_triangle_conservative,
            pipeline_triangle_regular,
            pipeline_upscale,
            pipeline_lines,
            bind_group_layout_upscale,
        }
    }

    fn resize(
        &mut self,
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let (low_res_target, bind_group_upscale) =
            Self::create_low_res_target(config, device, &self.bind_group_layout_upscale);
        self.low_res_target = low_res_target;
        self.bind_group_upscale = bind_group_upscale;
    }

    fn update(&mut self, _event: winit::event::WindowEvent) {}

    fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("primary"),
        });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("low resolution"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.low_res_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: &[],
            });

            rpass.set_pipeline(&self.pipeline_triangle_conservative);
            rpass.draw(0..3, 0..1);
            rpass.set_pipeline(&self.pipeline_triangle_regular);
            rpass.draw(0..3, 0..1);
        }
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("full resolution"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: &[],
            });

            rpass.set_pipeline(&self.pipeline_upscale);
            rpass.set_bind_group(0, &self.bind_group_upscale, &[]);
            rpass.draw(0..3, 0..1);

            if let Some(pipeline_lines) = &self.pipeline_lines {
                rpass.set_pipeline(pipeline_lines);
                rpass.draw(0..4, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("conservative-raster");
}

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[test]
#[wasm_bindgen_test::wasm_bindgen_test]
fn conservative_raster() {
    framework::test::<Example>(framework::FrameworkRefTest {
        image_path: "/examples/conservative-raster/screenshot.png",
        width: 1024,
        height: 768,
        optional_features: wgpu::Features::default(),
        base_test_parameters: framework::test_common::TestParameters::default(),
        tolerance: 0,
        max_outliers: 0,
    });
}
