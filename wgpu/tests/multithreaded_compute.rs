#[test]
#[cfg(any(feature = "vulkan", feature = "metal", feature = "dx12"))]
fn multithreaded_compute() {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    let thread_count = 8;

    let (tx, rx) = mpsc::channel();
    for _ in 0 .. thread_count {
        let tx = tx.clone();
        thread::spawn(move || {
            let numbers = vec![100, 100, 100];

            let size = (numbers.len() * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

            let instance = wgpu::Instance::new();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
            });

            let mut device = adapter.request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: wgpu::Limits::default(),
            });

            let cs = include_bytes!("../examples/hello-compute/shader.comp.spv");
            let cs_module = device
                .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&cs[..])).unwrap());

            let staging_buffer = device
                .create_buffer_mapped(
                    numbers.len(),
                    wgpu::BufferUsage::MAP_READ
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                )
                .fill_from_slice(&numbers);

            let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                size,
                usage: wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::COPY_SRC,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    bindings: &[wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageBuffer {
                            dynamic: false,
                            readonly: false,
                        },
                    }],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                bindings: &[wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &storage_buffer,
                        range: 0 .. size,
                    },
                }],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: &pipeline_layout,
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &cs_module,
                        entry_point: "main",
                    },
                });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
            encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
            {
                let mut cpass = encoder.begin_compute_pass();
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch(numbers.len() as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

            device.get_queue().submit(&[encoder.finish()]);

            staging_buffer.map_read_async(0, size, |result: wgpu::BufferMapAsyncResult<&[u32]>| {
                assert_eq!(result.unwrap().data, [25, 25, 25]);
            });
            tx.send(true).unwrap();
        });
    }

    for _ in 0 .. thread_count {
        rx.recv_timeout(Duration::from_secs(10))
            .expect("A thread never completed.");
    }
}
