#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]
use pollster::block_on;
use wgpu_benchmark::Benchmark;

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

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(base_backend),
            ..wgpu::InstanceDescriptor::from_env_or_default()
        });

        let adapter = block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .unwrap();

        let adapter_info = adapter.get_info();

        println!(
            "  Using adapter: {} ({:?})",
            adapter_info.name, adapter_info.backend
        );

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: adapter.features(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            label: None,
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        Self {
            adapter_info,
            device,
            queue,
        }
    }
}

fn main() {
    let benchmarks = vec![
        Benchmark {
            name: "Device::create_bind_group",
            func: bind_groups::run_bench,
        },
        Benchmark {
            name: "Device::create_buffer",
            func: resource_creation::run_bench,
        },
        Benchmark {
            name: "naga::front",
            func: shader::frontends,
        },
        Benchmark {
            name: "naga::valid",
            func: shader::validation,
        },
        Benchmark {
            name: "naga::compact",
            func: shader::compact,
        },
        Benchmark {
            name: "naga::back",
            func: shader::backends,
        },
        Benchmark {
            name: "Renderpass Encoding",
            func: renderpass::run_bench,
        },
        Benchmark {
            name: "Computepass Encoding",
            func: computepass::run_bench,
        },
    ];

    wgpu_benchmark::main(benchmarks);
}
