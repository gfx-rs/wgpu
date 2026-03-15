use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    vertex_attr_array,
};
use wgpu_test::{
    gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(SET_ARRAY_STRIDE_TO_0);
}

#[gpu_test]
static SET_ARRAY_STRIDE_TO_0: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .limits(wgpu::Limits::downlevel_defaults())
            // https://github.com/gfx-rs/wgpu/issues/9184
            .expect_fail(
                wgpu_test::FailureCase::molten_vk()
                    .validation_error("vertexAttributeAccessBeyondStride"),
            ),
    )
    .run_async(set_array_stride_to_0);

/// Tests that draws using a vertex buffer with stride of 0 works correctly (especially on the
/// D3D12 backend; see commentary within).
async fn set_array_stride_to_0(ctx: TestingContext) {
    let position_buffer_content: &[f32; 12] = &[
        // Triangle 1
        -1.0, -1.0, // Bottom left
        1.0, 1.0, // Top right
        -1.0, 1.0, // Top left
        // Triangle 2
        -1.0, -1.0, // Bottom left
        1.0, -1.0, // Bottom right
        1.0, 1.0, // Top right
    ];
    let position_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<f32, u8>(position_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let color_buffer_content: &[f32; 4] = &[1.0, 1.0, 1.0, 1.0];
    let color_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice::<f32, u8>(color_buffer_content),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let shader_src = "
        struct VertexOutput {
            @builtin(position) position: vec4f,
            @location(0) color: vec4f,
        }

        @vertex
        fn vs_main(@location(0) position: vec2f, @location(1) color: vec4f) -> VertexOutput {
            return VertexOutput(vec4f(position, 0.0, 1.0), color);
        }

        @fragment
        fn fs_main(@location(0) color: vec4f) -> @location(0) vec4f {
            return color;
        }
    ";

    let shader = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

    let vbl = [
        wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x2],
        },
        wgpu::VertexBufferLayout {
            array_stride: 0,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vertex_attr_array![1 => Float32x4],
        },
    ];
    let pipeline_desc = wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            buffers: &vbl,
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    };
    let mut first_pipeline_desc = pipeline_desc.clone();
    let mut first_vbl = vbl.clone();
    first_vbl[1].array_stride = 16;
    first_pipeline_desc.vertex.buffers = &first_vbl;
    let pipeline = ctx.device.create_render_pipeline(&pipeline_desc);
    let first_pipeline = ctx.device.create_render_pipeline(&first_pipeline_desc);

    let out_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_texture_view = out_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                ops: wgpu::Operations::default(),
                resolve_target: None,
                view: &out_texture_view,
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // The D3D12 backend used to not set the stride of vertex buffers if it was 0.
        rpass.set_pipeline(&first_pipeline); // This call caused the D3D12 backend to set the stride for the 2nd vertex buffer to 16.
        rpass.set_pipeline(&pipeline); // This call doesn't set the stride for the 2nd vertex buffer to 0.
        rpass.set_vertex_buffer(0, position_buffer.slice(..));
        rpass.set_vertex_buffer(1, color_buffer.slice(..));
        rpass.draw(0..6, 0..1); // Causing this draw to be skipped since it would read OOB of the 2nd vertex buffer.
    }

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &out_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit([encoder.finish()]);

    let slice = readback_buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| ());

    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();

    let data = slice.get_mapped_range();
    let succeeded = data.iter().all(|b| *b == u8::MAX);
    assert!(succeeded);
}
