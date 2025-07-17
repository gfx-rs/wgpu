/// To serve as an introduction to the wgpu api, we will implement a simple
/// compute shader which takes a list of numbers on the CPU and doubles them on the GPU.
///
/// While this isn't a very practical example, you will see all the major components
/// of using wgpu headlessly, including getting a device, running a shader, and transferring
/// data between the CPU and GPU.
///
/// If you time the recording and execution of this example you will certainly see that
/// running on the gpu is slower than doing the same calculation on the cpu. This is because
/// floating point multiplication is a very simple operation so the transfer/submission overhead
/// is quite a lot higher than the actual computation. This is normal and shows that the GPU
/// needs a lot higher work/transfer ratio to come out ahead.
use std::{num::NonZeroU64, str::FromStr};
use wgpu::util::DeviceExt;

fn main() {
    // Parse all arguments as floats. We need to skip argument 0, which is the name of the program.
    let arguments: Vec<f32> = std::env::args()
        .skip(1)
        .map(|s| {
            f32::from_str(&s).unwrap_or_else(|_| panic!("Cannot parse argument {s:?} as a float."))
        })
        .collect();

    if arguments.is_empty() {
        println!("No arguments provided. Please provide a list of numbers to double.");
        return;
    }

    println!("Parsed {} arguments", arguments.len());

    // wgpu uses `log` for all of our logging, so we initialize a logger with the `env_logger` crate.
    //
    // To change the log level, set the `RUST_LOG` environment variable. See the `env_logger`
    // documentation for more information.
    env_logger::init();

    // We first initialize an wgpu `Instance`, which contains any "global" state wgpu needs.
    //
    // This is what loads the vulkan/dx12/metal/opengl libraries.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    // We then create an `Adapter` which represents a physical gpu in the system. It allows
    // us to query information about it and create a `Device` from it.
    //
    // This function is asynchronous in WebGPU, so request_adapter returns a future. On native/webgl
    // the future resolves immediately, so we can block on it without harm.
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to create adapter");

    // Print out some basic information about the adapter.
    println!("Running on Adapter: {:#?}", adapter.get_info());

    // Check to see if the adapter supports compute shaders. While WebGPU guarantees support for
    // compute shaders, wgpu supports a wider range of devices through the use of "downlevel" devices.
    let downlevel_capabilities = adapter.get_downlevel_capabilities();
    if !downlevel_capabilities
        .flags
        .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
    {
        panic!("Adapter does not support compute shaders");
    }

    // We then create a `Device` and a `Queue` from the `Adapter`.
    //
    // The `Device` is used to create and manage GPU resources.
    // The `Queue` is a queue used to submit work for the GPU to process.
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .expect("Failed to create device");

    // Create a shader module from our shader code. This will parse and validate the shader.
    //
    // `include_wgsl` is a macro provided by wgpu like `include_str` which constructs a ShaderModuleDescriptor.
    // If you want to load shaders differently, you can construct the ShaderModuleDescriptor manually.
    let module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    // Create a buffer with the data we want to process on the GPU.
    //
    // `create_buffer_init` is a utility provided by `wgpu::util::DeviceExt` which simplifies creating
    // a buffer with some initial data.
    //
    // We use the `bytemuck` crate to cast the slice of f32 to a &[u8] to be uploaded to the GPU.
    let input_data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&arguments),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Now we create a buffer to store the output data.
    let output_data_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_data_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Finally we create a buffer which can be read by the CPU. This buffer is how we will read
    // the data. We need to use a separate buffer because we need to have a usage of `MAP_READ`,
    // and that usage can only be used with `COPY_DST`.
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: input_data_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // A bind group layout describes the types of resources that a bind group can contain. Think
    // of this like a C-style header declaration, ensuring both the pipeline and bind group agree
    // on the types of resources.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            // Input buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    has_dynamic_offset: false,
                },
                count: None,
            },
            // Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    // This is the size of a single element in the buffer.
                    min_binding_size: Some(NonZeroU64::new(4).unwrap()),
                    has_dynamic_offset: false,
                },
                count: None,
            },
        ],
    });

    // The bind group contains the actual resources to bind to the pipeline.
    //
    // Even when the buffers are individually dropped, wgpu will keep the bind group and buffers
    // alive until the bind group itself is dropped.
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_data_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_data_buffer.as_entire_binding(),
            },
        ],
    });

    // The pipeline layout describes the bind groups that a pipeline expects
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // The pipeline is the ready-to-go program state for the GPU. It contains the shader modules,
    // the interfaces (bind group layouts) and the shader entry point.
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("doubleMe"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // The command encoder allows us to record commands that we will later submit to the GPU.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    // A compute pass is a single series of compute operations. While we are recording a compute
    // pass, we cannot record to the encoder.
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    // Set the pipeline that we want to use
    compute_pass.set_pipeline(&pipeline);
    // Set the bind group that we want to use
    compute_pass.set_bind_group(0, &bind_group, &[]);

    // Now we dispatch a series of workgroups. Each workgroup is a 3D grid of individual programs.
    //
    // We defined the workgroup size in the shader as 64x1x1. So in order to process all of our
    // inputs, we ceiling divide the number of inputs by 64. If the user passes 32 inputs, we will
    // dispatch 1 workgroups. If the user passes 65 inputs, we will dispatch 2 workgroups, etc.
    let workgroup_count = arguments.len().div_ceil(64);
    compute_pass.dispatch_workgroups(workgroup_count as u32, 1, 1);

    // Now we drop the compute pass, giving us access to the encoder again.
    drop(compute_pass);

    // We add a copy operation to the encoder. This will copy the data from the output buffer on the
    // GPU to the download buffer on the CPU.
    encoder.copy_buffer_to_buffer(
        &output_data_buffer,
        0,
        &download_buffer,
        0,
        output_data_buffer.size(),
    );

    // We finish the encoder, giving us a fully recorded command buffer.
    let command_buffer = encoder.finish();

    // At this point nothing has actually been executed on the gpu. We have recorded a series of
    // commands that we want to execute, but they haven't been sent to the gpu yet.
    //
    // Submitting to the queue sends the command buffer to the gpu. The gpu will then execute the
    // commands in the command buffer in order.
    queue.submit([command_buffer]);

    // We now map the download buffer so we can read it. Mapping tells wgpu that we want to read/write
    // to the buffer directly by the CPU and it should not permit any more GPU operations on the buffer.
    //
    // Mapping requires that the GPU be finished using the buffer before it resolves, so mapping has a callback
    // to tell you when the mapping is complete.
    let buffer_slice = download_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {
        // In this case we know exactly when the mapping will be finished,
        // so we don't need to do anything in the callback.
    });

    // Wait for the GPU to finish working on the submitted work. This doesn't work on WebGPU, so we would need
    // to rely on the callback to know when the buffer is mapped.
    device.poll(wgpu::PollType::Wait).unwrap();

    // We can now read the data from the buffer.
    let data = buffer_slice.get_mapped_range();
    // Convert the data back to a slice of f32.
    let result: &[f32] = bytemuck::cast_slice(&data);

    // Print out the result.
    println!("Result: {result:?}");
}
