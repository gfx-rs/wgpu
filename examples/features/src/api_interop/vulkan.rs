#![allow(unused_variables, dead_code)]

type VkApi = wgpu::hal::vulkan::Api;

fn custom_instance_extensions() {
    // Storage for our custom vulkan extension. Actually fill this out in real code.
    let mut debug_report_create_info = ash::vk::DebugReportCallbackCreateInfoEXT::default();

    // Create a Vulkan wgpu-hal instance with custom extensions.
    let wgpu_hal_instance = unsafe {
        wgpu::hal::vulkan::Instance::init_with_callback(
            &wgpu::hal::InstanceDescriptor {
                name: "Vulkan Test",
                flags: wgpu::InstanceFlags::debugging(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                backend_options: wgpu::BackendOptions::default(),
            },
            // This callback allows you to add custom extensions and their feature flags without
            // wgpu's knowledge. Do not remove or disable any feature, wgpu will get mad.
            Some(Box::new(|args| {
                // Add a debug report extension.
                args.extensions.push(ash::ext::debug_report::NAME);
                // Extend the create info with the debug report create info.
                *args.create_info = args.create_info.push_next(&mut debug_report_create_info);
                // We also have access to the entry if we need it.
                let _ = args.entry;
            })),
        )
        .unwrap()
    };

    // Create a wgpu instance from the hal instance.
    let wgpu_instance = unsafe { wgpu::Instance::from_hal::<VkApi>(wgpu_hal_instance) };
}

fn custom_device_extensions(adapter: &wgpu::Adapter) {
    unsafe {
        // Storage for our custom device extensions. Actually fill this out in real code.
        let mut buffer_device_address_create_info =
            ash::vk::PhysicalDeviceBufferDeviceAddressFeatures::default();

        // Open an "OpenDevice" with the adapter, adding some extensions.
        let hal_device: wgpu::hal::OpenDevice<VkApi> = adapter
            .as_hal::<VkApi>()
            .unwrap()
            .open_with_callback(
                wgpu::Features::empty(),
                &wgpu::MemoryHints::Performance,
                Some(Box::new(|args| {
                    // Add the buffer device address extension.
                    args.extensions.push(ash::khr::buffer_device_address::NAME);
                    // Extend the create info with the buffer device address create info.
                    *args.create_info = args
                        .create_info
                        .push_next(&mut buffer_device_address_create_info);
                    // We also have access to the queue create infos if we need them.
                    let _ = args.queue_create_infos;
                })),
            )
            .unwrap();

        // Create a wgpu device from the hal device.
        let (device, queue) = adapter
            .create_device_from_hal(hal_device, &wgpu::DeviceDescriptor::default())
            .unwrap();
    }
}

fn as_hal() {
    unsafe {
        let instance: wgpu::Instance = unimplemented!();
        let adapter: wgpu::Adapter = unimplemented!();
        let device: wgpu::Device = unimplemented!();
        let queue: wgpu::Queue = unimplemented!();
        let buffer: wgpu::Buffer = unimplemented!();
        let texture: wgpu::Texture = unimplemented!();
        let view: wgpu::TextureView = unimplemented!();
        let sampler: wgpu::Sampler = unimplemented!();
        let shader_module: wgpu::ShaderModule = unimplemented!();
        let pipeline_layout: wgpu::PipelineLayout = unimplemented!();
        let render_pipeline: wgpu::RenderPipeline = unimplemented!();
        let compute_pipeline: wgpu::ComputePipeline = unimplemented!();
        let command_encoder: wgpu::CommandEncoder = unimplemented!();

        // Instance
        let hal_instance = instance.as_hal::<VkApi>().unwrap();
        hal_instance.shared_instance();
    }
}
