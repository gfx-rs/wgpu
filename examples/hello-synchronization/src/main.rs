const ARR_SIZE: usize = 128;

async fn run() {
    let mut local_patient_workgroup_results = [0_u32; ARR_SIZE];
    let mut local_hasty_workgroup_results = local_patient_workgroup_results;

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    execute(
        &device,
        &queue,
        &mut local_patient_workgroup_results[..],
        &mut local_hasty_workgroup_results[..],
    )
    .await;

    // Print data
    log::info!("Patient results: {local_patient_workgroup_results:?}");
    if !local_patient_workgroup_results.iter().any(|e| *e != 16) {
        log::info!("patient_main was patient.");
    } else {
        log::error!("patient_main was not patient!");
    }
    log::info!("Hasty results: {local_hasty_workgroup_results:?}");
    if local_hasty_workgroup_results.iter().any(|e| *e != 16) {
        log::info!("hasty_main was not patient.");
    } else {
        log::info!("hasty_main got lucky.");
    }
}

async fn execute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    local_patient_workgroup_results: &mut [u32],
    local_hasty_workgroup_results: &mut [u32],
) {
    let shaders_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders.wgsl"))),
    });

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(local_patient_workgroup_results) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(local_patient_workgroup_results) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let patient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shaders_module,
        entry_point: "patient_main",
    });
    let hasty_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shaders_module,
        entry_point: "hasty_main",
    });

    //----------------------------------------------------------

    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&patient_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(local_patient_workgroup_results.len() as u32, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    get_data(
        local_patient_workgroup_results,
        &storage_buffer,
        &output_staging_buffer,
        device,
        queue,
    )
    .await;

    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_pipeline(&hasty_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(local_patient_workgroup_results.len() as u32, 1, 1);
    }
    queue.submit(Some(command_encoder.finish()));

    get_data(
        local_hasty_workgroup_results,
        &storage_buffer,
        &output_staging_buffer,
        device,
        queue,
    )
    .await;
}

async fn get_data<T: bytemuck::Pod>(
    output: &mut [T],
    storage_buffer: &wgpu::Buffer,
    staging_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    command_encoder.copy_buffer_to_buffer(
        storage_buffer,
        0,
        staging_buffer,
        0,
        std::mem::size_of_val(output) as u64,
    );
    queue.submit(Some(command_encoder.finish()));
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap().unwrap();
    output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap();
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");

        web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.body())
            .expect("Could not get document / body.")
            .set_inner_html("<p>Nothing to see here! Open the console!</p>");

        wasm_bindgen_futures::spawn_local(run());
    }
}

#[cfg(test)]
mod tests;
