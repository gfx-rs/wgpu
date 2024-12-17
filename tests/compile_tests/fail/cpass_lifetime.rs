// Test to ensure that ComputePass without forget_lifetime does not compile
// when the ComputePass is dropped before the CommandBuffer is finished.
//
// See #6145 for more info.

fn main() {
    let instance = wgpu::Instance::new(Default::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

    let mut encoder = device.create_command_encoder(&Default::default());
    let _compute_pass = encoder.begin_compute_pass(&Default::default());
    // set up the compute pass...

    let cmd_buffer = encoder.finish();
    queue.submit([cmd_buffer]);
}
