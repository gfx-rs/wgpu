use hal;

use {binding_model, resource};

pub(crate) fn map_buffer_usage(
    usage: resource::BufferUsageFlags,
) -> (hal::buffer::Usage, hal::memory::Properties) {
    use hal::buffer::Usage as U;
    use hal::memory::Properties as P;
    use resource::BufferUsageFlags as W;

    let mut hal_memory = P::empty();
    if usage.contains(W::MAP_READ) {
        hal_memory |= P::CPU_VISIBLE | P::CPU_CACHED;
    }
    if usage.contains(W::MAP_WRITE) {
        hal_memory |= P::CPU_VISIBLE;
    }

    let mut hal_usage = U::empty();
    if usage.contains(W::TRANSFER_SRC) {
        hal_usage |= U::TRANSFER_SRC;
    }
    if usage.contains(W::TRANSFER_DST) {
        hal_usage |= U::TRANSFER_DST;
    }
    if usage.contains(W::INDEX) {
        hal_usage |= U::INDEX;
    }
    if usage.contains(W::VERTEX) {
        hal_usage |= U::VERTEX;
    }
    if usage.contains(W::UNIFORM) {
        hal_usage |= U::UNIFORM;
    }
    if usage.contains(W::STORAGE) {
        hal_usage |= U::STORAGE;
    }

    (hal_usage, hal_memory)
}

pub(crate) fn map_binding_type(
    binding_ty: &binding_model::BindingType,
) -> hal::pso::DescriptorType {
    use binding_model::BindingType::*;
    use hal::pso::DescriptorType as H;
    match binding_ty {
        UniformBuffer => H::UniformBuffer,
        Sampler => H::Sampler,
        SampledTexture => H::SampledImage,
        StorageBuffer => H::StorageBuffer,
    }
}

pub(crate) fn map_shader_stage_flags(
    shader_stage_flags: binding_model::ShaderStageFlags,
) -> hal::pso::ShaderStageFlags {
    use binding_model::{
        ShaderStageFlags_COMPUTE, ShaderStageFlags_FRAGMENT, ShaderStageFlags_VERTEX,
    };
    use hal::pso::ShaderStageFlags as H;
    let mut value = H::empty();
    if 0 != shader_stage_flags & ShaderStageFlags_VERTEX {
        value |= H::VERTEX;
    }
    if 0 != shader_stage_flags & ShaderStageFlags_FRAGMENT {
        value |= H::FRAGMENT;
    }
    if 0 != shader_stage_flags & ShaderStageFlags_COMPUTE {
        value |= H::COMPUTE;
    }
    value
}
