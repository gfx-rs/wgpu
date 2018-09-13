use hal;

pub struct CommandBuffer<B: hal::Backend> {
    raw: B::CommandBuffer,
}

pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
}

pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
}
