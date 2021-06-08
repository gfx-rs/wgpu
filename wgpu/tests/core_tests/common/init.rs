use std::panic::{catch_unwind, AssertUnwindSafe};

// Initialize the instance, obeying the WGPU_BACKEND environment variables.
pub fn get_backend_bits() -> wgpu::BackendBit {
    match std::env::var("WGPU_BACKEND")
        .as_deref()
        .map(str::to_lowercase)
        .as_deref()
    {
        Ok("vulkan") => wgpu::BackendBit::VULKAN,
        Ok("dx12") => wgpu::BackendBit::DX12,
        Ok("dx11") => wgpu::BackendBit::DX11,
        Ok("metal") => wgpu::BackendBit::METAL,
        Ok("gl") => wgpu::BackendBit::GL,
        Ok("webgpu") => wgpu::BackendBit::BROWSER_WEBGPU,
        Ok(_) => panic!("unknown wgpu backend"),
        Err(_) => wgpu::BackendBit::all(),
    }
}

// Initialize the adapter, obeying the WGPU_ADAPTER_NAME environment variable.
pub fn initialize_adapter(
    instance: &wgpu::Instance,
    backend_bits: wgpu::BackendBit,
) -> wgpu::Adapter {
    let adapters = instance.enumerate_adapters(backend_bits);

    let desired_adapter_name = std::env::var("WGPU_ADAPTER_NAME")
        .as_deref()
        .map(str::to_lowercase)
        .ok();

    let mut chosen_adapter = None;
    for adapter in adapters {
        let info = adapter.get_info();

        if chosen_adapter.is_none() {
            if let Some(ref adapter_name) = desired_adapter_name {
                if info.name.to_lowercase().contains(adapter_name) {
                    chosen_adapter = Some(adapter);
                    continue; // Skip adapter drop
                }
            } else {
                // Just choose first adapter if there's no preference
                chosen_adapter = Some(adapter);
                continue; // Skip adapter drop
            }
        }
    }

    chosen_adapter.expect("Could not find an adapter")
}

pub fn initialize_device(
    adapter: &wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
) -> (wgpu::Device, wgpu::Queue) {
    let bundle = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            features,
            limits,
        },
        None,
    ));

    match bundle {
        Ok(b) => b,
        Err(e) => panic!("Failed to initialize device: {}", e),
    }
}

pub struct TestingContext<'a> {
    pub adapter: &'a wgpu::Adapter,
    pub adapter_info: wgt::AdapterInfo,
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
}

// A rather arbitrary set of limits which should be lower than all devices wgpu reasonably expects to run on and provides enough resources for most tests to run.
// Adjust as needed if they are too low/high.
fn lowest_reasonable_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_1d: 512,
        max_texture_dimension_2d: 512,
        max_texture_dimension_3d: 32,
        max_texture_array_layers: 32,
        max_bind_groups: 1,
        max_dynamic_uniform_buffers_per_pipeline_layout: 1,
        max_dynamic_storage_buffers_per_pipeline_layout: 1,
        max_sampled_textures_per_shader_stage: 1,
        max_samplers_per_shader_stage: 1,
        max_storage_buffers_per_shader_stage: 1,
        max_storage_textures_per_shader_stage: 1,
        max_uniform_buffers_per_shader_stage: 1,
        max_uniform_buffer_binding_size: 256,
        max_storage_buffer_binding_size: 256,
        max_vertex_buffers: 1,
        max_vertex_attributes: 2,
        max_vertex_buffer_array_stride: 32,
        max_push_constant_size: 0,
    }
}

fn lowest_downlevel_properties() -> wgpu::DownlevelProperties {
    wgpu::DownlevelProperties {
        flags: wgt::DownlevelFlags::empty(),
        shader_model: wgt::ShaderModel::Sm2,
    }
}

// This information determines if a test should run.
pub struct TestParameters {
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
    pub required_downlevel_properties: wgpu::DownlevelProperties,
    // Test should always fail
    pub always_failure: bool,
    // Backends where test should fail.
    pub backend_failures: wgpu::BackendBit,
    // Vendors where test should fail.
    pub vendor_failures: &'static [usize],
    // Device names where test should fail.
    pub device_failures: &'static [&'static str],
}

impl Default for TestParameters {
    fn default() -> Self {
        Self {
            required_features: wgpu::Features::empty(),
            required_limits: lowest_reasonable_limits(),
            required_downlevel_properties: lowest_downlevel_properties(),
            always_failure: false,
            backend_failures: wgpu::BackendBit::empty(),
            vendor_failures: &[],
            device_failures: &[],
        }
    }
}

// Builder pattern to make it easier
impl TestParameters {
    pub fn features(mut self, features: wgpu::Features) -> Self {
        self.required_features &= features;
        self
    }
    
    pub fn failure(mut self) -> Self {
        self.always_failure = true;
        self
    }
}

pub fn initialize_test(
    parameters: TestParameters,
    test_function: impl FnOnce(&mut TestingContext<'_>),
) {
    // We don't actually care if it fails
    let _ = env_logger::try_init();

    let backend_bits = get_backend_bits();
    let instance = wgpu::Instance::new(backend_bits);
    let adapter = initialize_adapter(&instance, backend_bits);

    let adapter_info = adapter.get_info();
    let adapter_lowercase_name = adapter_info.name.to_lowercase();
    let adapter_features = adapter.features();
    let adapter_limits = adapter.limits();
    let adapter_downlevel_properties = adapter.get_downlevel_properties();

    let missing_features = parameters.required_features - adapter_features;
    if !missing_features.is_empty() {
        println!("TEST SKIPPED: MISSING FEATURES {:?}", missing_features);
        return;
    }

    if adapter_limits < parameters.required_limits {
        println!("TEST SKIPPED: LIMIT TOO LOW");
        return;
    }

    let missing_downlevel_flags =
        parameters.required_downlevel_properties.flags - adapter_downlevel_properties.flags;
    if !missing_downlevel_flags.is_empty() {
        println!(
            "TEST SKIPPED: MISSING DOWNLEVEL FLAGS {:?}",
            missing_downlevel_flags
        );
        return;
    }

    if adapter_downlevel_properties.shader_model
        < parameters.required_downlevel_properties.shader_model
    {
        println!(
            "TEST SKIPPED: LOW SHADER MODEL {:?}",
            adapter_downlevel_properties.shader_model
        );
        return;
    }

    let (device, queue) = initialize_device(
        &adapter,
        parameters.required_features,
        parameters.required_limits,
    );

    let mut context = TestingContext {
        adapter: &adapter,
        adapter_info: adapter_info.clone(),
        device: &device,
        queue: &queue,
    };

    let panicked = catch_unwind(AssertUnwindSafe(|| test_function(&mut context))).is_err();

    let expect_failure_backend = parameters
        .backend_failures
        .contains(wgt::BackendBit::from(adapter_info.backend));
    let expect_failure_vendor = parameters
        .vendor_failures
        .iter()
        .find(|&&v| v == adapter_info.vendor)
        .is_some();
    let expect_failure_device = parameters
        .device_failures
        .iter()
        .find(|&&v| adapter_lowercase_name.contains(&v.to_lowercase()));

    let expect_failure = parameters.always_failure
        || expect_failure_backend
        || expect_failure_vendor
        || expect_failure_device.is_some();

    if panicked == expect_failure {
        // We got the conditions we expected
        if expect_failure {
            // Print out reason for the failure
            if parameters.always_failure {
                println!("GOT EXPECTED TEST FAILURE: ALWAYS");
            }
            if expect_failure_backend {
                println!(
                    "GOT EXPECTED TEST FAILURE: BACKEND {:?}",
                    adapter_info.backend
                );
            }
            if expect_failure_vendor {
                println!("GOT EXPECTED TEST FAILURE: VENDOR {}", adapter_info.vendor);
            }
            if let Some(device_match) = expect_failure_device {
                println!(
                    "GOT EXPECTED TEST FAILURE: DEVICE {} MATCHED NAME {}",
                    adapter_info.name, device_match
                );
            }
        }
    } else {
        if expect_failure {
            // We expected to fail, but things passed
            if parameters.always_failure {
                panic!("UNEXPECTED TEST PASS: ALWAYS");
            }
            if expect_failure_backend {
                panic!("UNEXPECTED TEST PASS: BACKEND {:?}", adapter_info.backend);
            }
            if expect_failure_vendor {
                panic!("UNEXPECTED TEST PASS: VENDOR {}", adapter_info.vendor);
            }
            if let Some(device_match) = expect_failure_device {
                panic!(
                    "UNEXPECTED TEST PASS: DEVICE {} MATCHED NAME {}",
                    adapter_info.name, device_match
                );
            }
            unreachable!()
        } else {
            panic!("UNEXPECTED TEST FAILURE")
        }
    }
}
