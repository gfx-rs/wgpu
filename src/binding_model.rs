use hal;


pub struct PipelineLayout<B: hal::Backend> {
    raw: B::PipelineLayout,
}
