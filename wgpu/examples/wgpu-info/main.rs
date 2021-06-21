use std::{
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
    println!("\t\tDepth Clamping:                             {}", features.contains(wgpu::Features::DEPTH_CLAMPING));
    println!("\t\tTexture Compression BC:                     {}", features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC));
    println!("\t\tTimestamp Query:                            {}", features.contains(wgpu::Features::TIMESTAMP_QUERY));
    println!("\t\tPipeline Statistics Query:                  {}", features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY));
    println!("\t\tMappable Primary Buffers:                   {}", features.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS));
    println!("\t\tSampled Texture Binding Array:              {}", features.contains(wgpu::Features::SAMPLED_TEXTURE_BINDING_ARRAY));
    println!("\t\tSampled Texture Array Dynamic Indexing:     {}", features.contains(wgpu::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING));
    println!("\t\tSampled Texture Array Non Uniform Indexing: {}", features.contains(wgpu::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING));
    println!("\t\tUnsized Binding Array:                      {}", features.contains(wgpu::Features::UNSIZED_BINDING_ARRAY));
    println!("\t\tMulti Draw Indirect:                        {}", features.contains(wgpu::Features::MULTI_DRAW_INDIRECT));
    println!("\t\tMulti Draw Indirect Count:                  {}", features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT));
    println!("\t\tPush Constants:                             {}", features.contains(wgpu::Features::PUSH_CONSTANTS));
    println!("\t\tAddress Mode Clamp To Border:               {}", features.contains(wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER));
    println!("\t\tNon Fill Polygon Mode:                      {}", features.contains(wgpu::Features::NON_FILL_POLYGON_MODE));
    println!("\t\tTexture Compression ETC2:                   {}", features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2));
    println!("\t\tTexture Compression ASTC LDR:               {}", features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR));
    println!("\t\tTexture Adapter Specific Format Features:   {}", features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES));
    println!("\t\tShader Float64:                             {}", features.contains(wgpu::Features::SHADER_FLOAT64));
    println!("\t\tVertex Attribute 64bit:                     {}", features.contains(wgpu::Features::VERTEX_ATTRIBUTE_64BIT));
    println!("\t\tConservative Rasterization:                 {}", features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION));
    println!("\t\tBuffer Binding Array:                       {}", features.contains(wgpu::Features::BUFFER_BINDING_ARRAY));
    println!("\t\tUniform Buffer Array Dynamic Indexing:      {}", features.contains(wgpu::Features::UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING));
    println!("\t\tUniform Buffer Array Non Uniform Indexing:  {}", features.contains(wgpu::Features::UNIFORM_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
    println!("\t\tStorage Buffer Array Non Uniform Indexing:  {}", features.contains(wgpu::Features::STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING));
    println!("\t\tVertex Writable Storage:                    {}", features.contains(wgpu::Features::VERTEX_WRITABLE_STORAGE));
    println!("\t\tClear Commands:                             {}", features.contains(wgpu::Features::CLEAR_COMMANDS));
    println!("\tLimits:");
    println!("\t\tMax Texture Dimension 1d:                        {}", limits.max_texture_dimension_1d);
    println!("\t\tMax Texture Dimension 2d:                        {}", limits.max_texture_dimension_2d);
    println!("\t\tMax Texture Dimension 3d:                        {}", limits.max_texture_dimension_3d);
    println!("\t\tMax Texture Array Layers:                        {}", limits.max_texture_array_layers);
    println!("\t\tMax Bind Groups:                                 {}", limits.max_bind_groups);
    println!("\t\tMax Dynamic Uniform Buffers Per Pipeline Layout: {}", limits.max_dynamic_uniform_buffers_per_pipeline_layout);
    println!("\t\tMax Dynamic Storage Buffers Per Pipeline Layout: {}", limits.max_dynamic_storage_buffers_per_pipeline_layout);
    println!("\t\tMax Sampled Textures Per Shader Stage:           {}", limits.max_sampled_textures_per_shader_stage);
    println!("\t\tMax Samplers Per Shader Stage:                   {}", limits.max_samplers_per_shader_stage);
    println!("\t\tMax Storage Buffers Per Shader Stage:            {}", limits.max_storage_buffers_per_shader_stage);
    println!("\t\tMax Storage Textures Per Shader Stage:           {}", limits.max_storage_textures_per_shader_stage);
    println!("\t\tMax Uniform Buffers Per Shader Stage:            {}", limits.max_uniform_buffers_per_shader_stage);
    println!("\t\tMax Uniform Buffer Binding Size:                 {}", limits.max_uniform_buffer_binding_size);
    println!("\t\tMax Storage Buffer Binding Size:                 {}", limits.max_storage_buffer_binding_size);
    println!("\t\tMax Vertex Buffers:                              {}", limits.max_vertex_buffers);
    println!("\t\tMax Vertex Attributes:                           {}", limits.max_vertex_attributes);
    println!("\t\tMax Vertex Buffer Array Stride:                  {}", limits.max_vertex_buffer_array_stride);
    println!("\t\tMax Push Constant Size:                          {}", limits.max_push_constant_size);
    println!("\tDownlevel Properties:");
    println!("\t\tShader Model:                        {:?}", downlevel.shader_model);
    println!("\t\tCompute Shaders:                     {}", downlevel.flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS));
    println!("\t\tStorage Images:                      {}", downlevel.flags.contains(wgpu::DownlevelFlags::STORAGE_IMAGES));
    println!("\t\tRead Only Depth Stencil:             {}", downlevel.flags.contains(wgpu::DownlevelFlags::READ_ONLY_DEPTH_STENCIL));
    println!("\t\tDevice Local Image Copies:           {}", downlevel.flags.contains(wgpu::DownlevelFlags::DEVICE_LOCAL_IMAGE_COPIES));
    println!("\t\tNon Power Of Two Mipmapped Textures: {}", downlevel.flags.contains(wgpu::DownlevelFlags::NON_POWER_OF_TWO_MIPMAPPED_TEXTURES));
    println!("\t\tCube Array Textures:                 {}", downlevel.flags.contains(wgpu::DownlevelFlags::CUBE_ARRAY_TEXTURES));
    println!("\t\tAnisotropic Filtering:               {}", downlevel.flags.contains(wgpu::DownlevelFlags::ANISOTROPIC_FILTERING));
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();

    let instance = wgpu::Instance::new(wgpu::BackendBit::all());
    let adapters: Vec<_> = instance
        .enumerate_adapters(wgpu::BackendBit::all())
        .collect();
    let adapter_count = adapters.len();

    let all_start = Instant::now();

    if args.is_empty() {
        for (idx, adapter) in adapters.into_iter().enumerate() {
            print_info_from_adapter(&adapter, idx)
        }
    } else {
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
    }

    let all_time = all_start.elapsed().as_secs_f32();

    println!(
        "=========== {} adapters PASSED in {:.3}s ===========",
        adapter_count, all_time
    );
}
