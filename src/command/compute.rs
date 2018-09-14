use hal;

pub struct ComputePass<B: hal::Backend> {
    raw: B::CommandBuffer,
}
