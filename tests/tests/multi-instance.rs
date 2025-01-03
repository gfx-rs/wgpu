#![cfg(not(target_arch = "wasm32"))]

async fn get() -> wgpu::Adapter {
    let adapter = {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::util::backend_bits_from_env().unwrap_or_default(),
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
fn multi_instance() {
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
