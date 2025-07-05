mod multi_instance {
    #![cfg(not(target_arch = "wasm32"))]

    async fn get() -> wgpu::Adapter {
        let adapter = {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::from_env().unwrap_or_default(),
                ..Default::default()
            });
            instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap()
        };

        log::info!("Selected adapter: {:?}", adapter.get_info());

        adapter
    }

    #[test]
    pub fn multi_instance() {
        {
            env_logger::init();

            // Sequential instances.
            for _ in 0..3 {
                pollster::block_on(get());
            }

            // Concurrent instances
            let _instances: Vec<_> = (0..3).map(|_| pollster::block_on(get())).collect();
        }
    }
}

mod request_adapter_error {
    fn id(backends: wgpu::Backends) -> wgpu::InstanceDescriptor {
        wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
        }
    }

    fn adapter_error(desc: &wgpu::InstanceDescriptor) -> String {
        let instance = wgpu::Instance::new(desc);
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .unwrap_err()
        .to_string()
    }

    #[test]
    fn no_backends_requested() {
        assert_eq!(
            adapter_error(&id(wgpu::Backends::empty())),
            "No suitable graphics adapter found; \
            noop not requested, \
            vulkan not requested, \
            metal not requested, \
            dx12 not requested, \
            gl not requested, \
            webgpu not requested"
        );
    }

    /// This test nominally tests the noop backend, but it also exercises the logic for other backends
    /// that fail to return any adapter.
    #[test]
    fn noop_not_enabled() {
        assert_eq!(
            adapter_error(&id(wgpu::Backends::NOOP)),
            "No suitable graphics adapter found; \
            noop not explicitly enabled, \
            vulkan not requested, \
            metal not requested, \
            dx12 not requested, \
            gl not requested, \
            webgpu not requested"
        );
    }

    #[test]
    fn no_compiled_support() {
        // Whichever platform we are on, try asking for a backend that definitely will be
        // cfged out regardless of feature flags. (Not that these tests run on wasm at all yet.)

        #[cfg(target_family = "wasm")]
        assert_eq!(
            adapter_error(&id(wgpu::Backends::METAL)),
            "No suitable graphics adapter found; \
            noop not requested, \
            vulkan not requested, \
            metal support not compiled in, \
            dx12 not requested, \
            gl not requested, \
            webgpu not requested"
        );

        #[cfg(not(target_family = "wasm"))]
        assert_eq!(
            adapter_error(&id(wgpu::Backends::BROWSER_WEBGPU)),
            "No suitable graphics adapter found; \
            noop not requested, \
            vulkan not requested, \
            metal not requested, \
            dx12 not requested, \
            gl not requested, \
            webgpu support not compiled in"
        );
    }
}
