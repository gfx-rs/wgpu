use hal;

use {CommandBuffer, CommandBufferHandle, ComputePassHandle};


pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
}

pub extern "C"
fn compute_pass_dispatch(
    pass: ComputePassHandle, groups_x: u32, groups_y: u32, groups_z: u32
) {
    unimplemented!()
}

pub extern "C"
fn compute_pass_end(pass: ComputePassHandle) -> CommandBufferHandle {
    match pass.unbox() {
        Some(pass) => CommandBufferHandle::new(CommandBuffer {
            raw: pass.raw,
        }),
        None => CommandBufferHandle::null(),
    }
}
