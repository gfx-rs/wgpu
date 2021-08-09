use std::{
    mem::size_of,
    process::{exit, Command},
    time::Instant,
};

// Lets keep these on one line
#[rustfmt::skip]
fn print_info_from_adapter(adapter: &wgpu::Adapter, idx: usize) {
    let info = adapter.get_info();
    let downlevel = adapter.get_downlevel_properties();
    let features = adapter.features();
    let limits = adapter.limits();

    println!("Adapter {}:", idx);
    println!("\tBackend:   {:?}", info.backend);
    println!("\tName:      {:?}", info.name);
    println!("\tVendorID:  {:?}", info.vendor);
    println!("\tDeviceID:  {:?}", info.device);
    println!("\tType:      {:?}", info.device_type);
    println!("\tCompliant: {:?}", downlevel.is_webgpu_compliant());
    println!("\tFeatures:");
    for i in 0..(size_of::<wgpu::Features>() * 8) {
        let bit = wgpu::Features::from_bits(1 << i as u64);
        if let Some(bit) = bit {
            if wgpu::Features::all().contains(bit) {
                println!("\t\t{:<63} {}", format!("{:?}:", bit), features.contains(bit));
            }
        }
    }
    println!("\tLimits:");
    let wgpu::Limits { 
        max_texture_dimension_1d, 
        max_texture_dimension_2d, 
        max_texture_dimension_3d, 
        max_texture_array_layers, 
        max_bind_groups, 
        max_dynamic_uniform_buffers_per_pipeline_layout, 
        max_dynamic_storage_buffers_per_pipeline_layout, 
        max_sampled_textures_per_shader_stage, 
        max_samplers_per_shader_stage, 
        max_storage_buffers_per_shader_stage, 
        max_storage_textures_per_shader_stage, 
        max_uniform_buffers_per_shader_stage, 
        max_uniform_buffer_binding_size, 
        max_storage_buffer_binding_size, 
        max_vertex_buffers, 
        max_vertex_attributes, 
        max_vertex_buffer_array_stride, 
        max_push_constant_size, 
    } = limits;
    println!("\t\tMax Texture Dimension 1d:                        {}", max_texture_dimension_1d);
    println!("\t\tMax Texture Dimension 2d:                        {}", max_texture_dimension_2d);
    println!("\t\tMax Texture Dimension 3d:                        {}", max_texture_dimension_3d);
    println!("\t\tMax Texture Array Layers:                        {}", max_texture_array_layers);
    println!("\t\tMax Bind Groups:                                 {}", max_bind_groups);
    println!("\t\tMax Dynamic Uniform Buffers Per Pipeline Layout: {}", max_dynamic_uniform_buffers_per_pipeline_layout);
    println!("\t\tMax Dynamic Storage Buffers Per Pipeline Layout: {}", max_dynamic_storage_buffers_per_pipeline_layout);
    println!("\t\tMax Sampled Textures Per Shader Stage:           {}", max_sampled_textures_per_shader_stage);
    println!("\t\tMax Samplers Per Shader Stage:                   {}", max_samplers_per_shader_stage);
    println!("\t\tMax Storage Buffers Per Shader Stage:            {}", max_storage_buffers_per_shader_stage);
    println!("\t\tMax Storage Textures Per Shader Stage:           {}", max_storage_textures_per_shader_stage);
    println!("\t\tMax Uniform Buffers Per Shader Stage:            {}", max_uniform_buffers_per_shader_stage);
    println!("\t\tMax Uniform Buffer Binding Size:                 {}", max_uniform_buffer_binding_size);
    println!("\t\tMax Storage Buffer Binding Size:                 {}", max_storage_buffer_binding_size);
    println!("\t\tMax Vertex Buffers:                              {}", max_vertex_buffers);
    println!("\t\tMax Vertex Attributes:                           {}", max_vertex_attributes);
    println!("\t\tMax Vertex Buffer Array Stride:                  {}", max_vertex_buffer_array_stride);
    println!("\t\tMax Push Constant Size:                          {}", max_push_constant_size);
    println!("\tDownlevel Properties:");
    let wgpu::DownlevelCapabilities {
        shader_model,
        limits: _,
        flags,
    } = downlevel;
    println!("\t\tShader Model:                        {:?}", shader_model);
    for i in 0..(size_of::<wgpu::DownlevelFlags>() * 8) {
        let bit = wgpu::DownlevelFlags::from_bits(1 << i as u64);
        if let Some(bit) = bit {
            if wgpu::DownlevelFlags::all().contains(bit) {
                println!("\t\t{:<36} {}", format!("{:?}:", bit), flags.contains(bit));
            }
        }
    }
}

fn main() {
    env_logger::init();

    let args: Vec<_> = std::env::args().skip(1).collect();

    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).collect();
    let adapter_count = adapters.len();

    if args.is_empty() {
        for (idx, adapter) in adapters.into_iter().enumerate() {
            print_info_from_adapter(&adapter, idx)
        }
    } else {
        let all_start = Instant::now();

        for (idx, adapter) in adapters.into_iter().enumerate() {
            let adapter_start_time = Instant::now();
            let idx = idx + 1;
            let info = adapter.get_info();
            println!(
                "=========== TESTING {} on {:?} ({} of {}) ===========",
                info.name, info.backend, idx, adapter_count
            );
            let exit_status = Command::new(&args[0])
                .args(&args[1..])
                .env("WGPU_ADAPTER_NAME", &info.name)
                .env(
                    "WGPU_BACKEND",
                    match info.backend {
                        wgpu::Backend::Empty => unreachable!(),
                        wgpu::Backend::Vulkan => "vulkan",
                        wgpu::Backend::Metal => "metal",
                        wgpu::Backend::Dx12 => "dx12",
                        wgpu::Backend::Dx11 => "dx11",
                        wgpu::Backend::Gl => "gl",
                        wgpu::Backend::BrowserWebGpu => "webgpu",
                    },
                )
                .spawn()
                .unwrap()
                .wait()
                .unwrap();

            let adapter_time = adapter_start_time.elapsed().as_secs_f32();

            if exit_status.success() {
                println!(
                    "=========== PASSED! {} on {:?} ({} of {}) in {:.3}s ===========",
                    info.name, info.backend, idx, adapter_count, adapter_time
                );
            } else {
                println!(
                    "=========== FAILED! {} on {:?} ({} of {}) in {:.3}s ===========",
                    info.name, info.backend, idx, adapter_count, adapter_time
                );
                exit(1);
            }
        }

        let all_time = all_start.elapsed().as_secs_f32();

        println!(
            "=========== {} adapters PASSED in {:.3}s ===========",
            adapter_count, all_time
        );
    }
}
