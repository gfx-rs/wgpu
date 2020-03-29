use std::{convert::TryInto as _, str::FromStr};
use zerocopy::AsBytes as _;

async fn run() {
    let numbers = if std::env::args().len() == 1 {
        let default = vec![1, 2, 3, 4];
        log::info!("No numbers were provided, defaulting to {:?}", default);
        default
    } else {
        std::env::args()
            .skip(1)
            .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
            .collect()
    };

    // To see the output, run `RUST_LOG=info cargo run --example hello-compute`.
    log::info!("Times: {:?}", execute_gpu(numbers).await);
}

async fn execute_gpu(numbers: Vec<u32>) -> Vec<u32> {
    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    let adapter = wgpu::Adapter::request(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
        },
        wgpu::BackendBit::PRIMARY,
    )
    .await
    .unwrap();

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
        limits: wgpu::Limits::default(),
    })
    .await;

    let cs = include_bytes!("shader.comp.spv");
    let cs_module =
        device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&cs[..])).unwrap());

    let staging_buffer = device.create_buffer_with_data(
        numbers.as_slice().as_bytes(),
        wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST | wgpu::BufferUsage::COPY_SRC,
    );

    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size,
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        bindings: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
            },
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[wgpu::Binding {
            binding: 0,
            resource: wgpu::BindingResource::Buffer {
                buffer: &storage_buffer,
                range: 0 .. size,
            },
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        layout: &pipeline_layout,
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { todo: 0 });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, &storage_buffer, 0, size);
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(numbers.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    queue.submit(&[encoder.finish()]);

    // Note that we're not calling `.await` here.
    let buffer_future = staging_buffer.map_read(0, size);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(mapping) = buffer_future.await {
        mapping
            .as_slice()
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    env_logger::init();

    futures::executor::block_on(run());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_1(){
        let input = vec!(1, 2, 3, 4);
        futures::executor::block_on(assert_execute_gpu(input, vec!(0, 1, 7, 2)));
    }

    #[test]
    fn test_compute_2(){
        let input = vec!(5, 23, 10, 9);
        futures::executor::block_on(assert_execute_gpu(input, vec!(5, 15, 6, 19)));
    }

    #[test]
    fn test_multithreaded_compute() {
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let thread_count = 8;

        let (tx, rx) = mpsc::channel();
        for _ in 0 .. thread_count {
            let tx = tx.clone();
            thread::spawn(move || {
                let input = vec![100, 100, 100];
                futures::executor::block_on(assert_execute_gpu(input, vec!(25, 25, 25)));
                tx.send(true).unwrap();
            });
        }

        for _ in 0 .. thread_count {
            rx.recv_timeout(Duration::from_secs(10))
                .expect("A thread never completed.");
        }
    }

    async fn assert_execute_gpu(input: Vec<u32>, expected: Vec<u32>) {
        assert_eq!(execute_gpu(input).await, expected);
    }
}
