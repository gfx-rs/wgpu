use crate::common::{image::calc_difference, initialize_test, TestParameters, TestingContext};
use wgpu::{util::DeviceExt, DownlevelFlags, Limits, TextureFormat};

#[test]
fn reinterpret_srgb_ness() {
    let parameters = TestParameters::default()
        .downlevel_flags(DownlevelFlags::VIEW_FORMATS)
        .limits(Limits::downlevel_defaults())
        .specific_failure(Some(wgpu::Backends::GL), None, None, true);
    initialize_test(parameters, |ctx| {
        let unorm_data: [[u8; 4]; 4] = [
            [180, 0, 0, 255],
            [0, 84, 0, 127],
            [0, 0, 62, 100],
            [62, 180, 84, 90],
        ];
        let srgb_data: [[u8; 4]; 4] = [
            [116, 0, 0, 255],
            [0, 23, 0, 127],
            [0, 0, 12, 100],
            [12, 116, 23, 90],
        ];

        let size = wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        };

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("view_format.wgsl"));

        // Reinterpret Rgba8Unorm as Rgba8UnormSrgb
        reinterpret(
            &ctx,
            &shader,
            size,
            TextureFormat::Rgba8Unorm,
            TextureFormat::Rgba8UnormSrgb,
            &unorm_data,
            &srgb_data,
        );

        // Reinterpret Rgba8UnormSrgb back to Rgba8Unorm
        reinterpret(
            &ctx,
            &shader,
            size,
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8Unorm,
            &srgb_data,
            &unorm_data,
        );
    });
}

fn reinterpret(
    ctx: &TestingContext,
    shader: &wgpu::ShaderModule,
    size: wgpu::Extent3d,
    src_format: wgpu::TextureFormat,
    reinterpret_to: wgpu::TextureFormat,
    src_data: &[[u8; 4]],
    expect_data: &[[u8; 4]],
) {
    let tex = ctx.device.create_texture_with_data(
        &ctx.queue,
        &wgpu::TextureDescriptor {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            size,
            format: src_format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[reinterpret_to],
        },
        bytemuck::cast_slice(src_data),
    );
    let tv = tex.create_view(&wgpu::TextureViewDescriptor {
        format: Some(reinterpret_to),
        ..Default::default()
    });
    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reinterpret pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(src_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&tv),
        }],
        label: None,
    });

    let target_tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: src_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            ops: wgpu::Operations::default(),
            resolve_target: None,
            view: &target_view,
        })],
        depth_stencil_attachment: None,
        label: None,
        timestamp_writes: &[],
    });
    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bind_group, &[]);
    rpass.draw(0..3, 0..1);
    drop(rpass);
    ctx.queue.submit(Some(encoder.finish()));

    let read_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64 * 2,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &target_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &read_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: None,
            },
        },
        size,
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = read_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());
    ctx.device.poll(wgpu::Maintain::Wait);

    let data: Vec<u8> = slice.get_mapped_range().to_vec();
    let tolerance_data: [[u8; 4]; 4] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]];

    for h in 0..size.height {
        let offset = h * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        for w in 0..size.width {
            let expect = expect_data[(h * size.width + w) as usize];
            let tolerance = tolerance_data[(h * size.width + w) as usize];
            let index = (w * 4 + offset) as usize;
            if calc_difference(expect[0], data[index]) > tolerance[0]
                || calc_difference(expect[1], data[index + 1]) > tolerance[1]
                || calc_difference(expect[2], data[index + 2]) > tolerance[2]
                || calc_difference(expect[3], data[index + 3]) > tolerance[3]
            {
                panic!(
                    "Reinterpret {:?} as {:?} mismatch! expect {:?} get [{}, {}, {}, {}]",
                    src_format,
                    reinterpret_to,
                    expect,
                    data[index],
                    data[index + 1],
                    data[index + 2],
                    data[index + 3]
                )
            }
        }
    }
}
