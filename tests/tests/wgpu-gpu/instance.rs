use wgpu_test::{apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(INITIALIZE);
    #[cfg(target_vendor = "apple")]
    vec.push(METAL_MSL_VERSION_OPTIONS);
}

#[apply(gpu_test!)]
static INITIALIZE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().enable_noop())
    .run_sync(|_ctx| {});

#[cfg(target_vendor = "apple")]
#[apply(gpu_test!)]
static METAL_MSL_VERSION_OPTIONS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().skip(wgpu_test::FailureCase::backend(!wgpu::Backends::METAL)),
    )
    .run_async(|ctx| async move {
        use wgpu::{MetalBackendOptions, MetalShadingLanguageVersion as Version};

        for options in [
            MetalBackendOptions::default(),
            MetalBackendOptions {
                msl_version: Some(Version::Version2_0),
                ..Default::default()
            },
            MetalBackendOptions {
                max_msl_version: Some(Version::Version2_0),
                ..Default::default()
            },
            MetalBackendOptions {
                msl_version: Some(Version::Version2_0),
                max_msl_version: Some(Version::Version1_2),
            },
        ] {
            let started = std::time::Instant::now();
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                backend_options: wgpu::BackendOptions {
                    metal: options.clone(),
                    ..Default::default()
                },
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapter = instance
                .enumerate_adapters(wgpu::Backends::METAL)
                .await
                .into_iter()
                .find(|adapter| adapter.get_info().name == ctx.adapter.get_info().name)
                .expect("the current Metal adapter should support MSL 2.0");
            println!("Metal discovery {options:?}: {:?}", started.elapsed());
            if options != MetalBackendOptions::default() {
                assert!(!adapter.features().intersects(
                    wgpu::Features::TEXTURE_ATOMIC
                        | wgpu::Features::MEMORY_DECORATION_COHERENT
                        | wgpu::Features::DEBUG_PRINTF
                        | wgpu::Features::EXPERIMENTAL_MESH_SHADER
                        | wgpu::Features::SHADER_PER_VERTEX
                        | wgpu::Features::PRIMITIVE_INDEX
                        | wgpu::Features::SHADER_BARYCENTRICS
                        | wgpu::Features::MULTIVIEW
                        | wgpu::Features::EXPERIMENTAL_RAY_QUERY
                ));
                assert!(!adapter
                    .get_downlevel_capabilities()
                    .flags
                    .contains(wgpu::DownlevelFlags::MSL2_1));
            }
            let (device, _queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("MSL version regression"),
                source: wgpu::ShaderSource::Wgsl(
                    "@compute @workgroup_size(1) fn cs() {}\n\
                     @vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n\
                     @fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
                        .into(),
                ),
            });
            let _ = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: Some("cs"),
                compilation_options: Default::default(),
                cache: None,
            });
            let _ = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                }),
                multiview_mask: None,
                cache: None,
            });
        }
    });
