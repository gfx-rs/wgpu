use criterion::criterion_main;
use pollster::block_on;

mod bind_groups;
mod computepass;
mod renderpass;
mod resource_creation;
mod shader;

struct DeviceState {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl DeviceState {
    fn new() -> Self {
        #[cfg(feature = "tracy")]
        tracy_client::Client::start();

        let base_backend = if cfg!(target_os = "macos") {
            // We don't want to use Molten-VK on Mac.
            wgpu::Backends::METAL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(base_backend),
            ..wgpu::InstanceDescriptor::from_env_or_default()
        });

        let adapter = block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .unwrap();

        let adapter_info = adapter.get_info();

        eprintln!("{adapter_info:?}");

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: adapter.features(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("Compute/RenderPass Device"),
            },
            None,
        ))
        .unwrap();

        Self {
            adapter_info,
            device,
            queue,
        }
    }
}

criterion_main!(
    bind_groups::bind_groups,
    renderpass::renderpass,
    computepass::computepass,
    resource_creation::resource_creation,
    shader::shader
);
