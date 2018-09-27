use hal;

//use {CommandBuffer, CommandBufferId, ComputePassId};

pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
}
