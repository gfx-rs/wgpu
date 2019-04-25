extern crate env_logger;
extern crate wgpu;
extern crate wgpu_native;

use std::str::FromStr;

fn main() {
    env_logger::init();

    // For now this just panics if you didn't pass numbers. Could add proper error handling.
    if std::env::args().len() == 1 {
        panic!("You must pass a list of positive integers!")
    }
    let numbers: Vec<u32> = std::env::args()
        .skip(1)
        .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
        .collect();

    let size = (numbers.len() * std::mem::size_of::<u32>()) as u32;

    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::Default,
    });
    let mut device = adapter.create_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
    });

    let cs_bytes = include_bytes!("./../data/collatz.comp.spv");
    let cs_module = device.create_shader_module(cs_bytes);

    let staging_buffer = device
        .create_buffer_mapped(
            numbers.len(),
            wgpu::BufferUsageFlags::MAP_READ
                | wgpu::BufferUsageFlags::TRANSFER_DST
                | wgpu::BufferUsageFlags::TRANSFER_SRC,
        )
        .fill_from_slice(&numbers);

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsageFlags::STORAGE
            | wgpu::BufferUsageFlags::TRANSFER_DST
            | wgpu::BufferUsageFlags::TRANSFER_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[wgpu::BindGroupLayoutBinding {
            binding: 0,
            visibility: wgpu::ShaderStageFlags::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer {
                buffer: &storage_buffer,
                range: 0..size,
            },
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::PipelineStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group);
        cpass.dispatch(numbers.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    device.get_queue().submit(&[encoder.finish()]);

    staging_buffer.map_read_async(0, size, |result: wgpu::BufferMapAsyncResult<&[u32]>| {
        if let Ok(mapping) = result {
            println!("Times: {:?}", mapping.data);
        }
    });
}
