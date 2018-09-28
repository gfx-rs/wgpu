use hal;

//use {CommandBuffer, CommandBufferId, RenderPassId};

pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
}
