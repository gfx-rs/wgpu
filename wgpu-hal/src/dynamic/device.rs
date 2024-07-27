// Box casts are needed, alternative would be a temporaries which are more verbose and not more expressive.
#![allow(trivial_casts)]

use crate::{
    Api, BindGroupDescriptor, BindGroupLayoutDescriptor, BufferDescriptor, BufferMapping,
    CommandEncoderDescriptor, Device, DeviceError, DynBuffer, DynResource, MemoryRange,
    PipelineLayoutDescriptor, SamplerDescriptor, ShaderError, ShaderInput, ShaderModuleDescriptor,
    TextureDescriptor, TextureViewDescriptor,
};

use super::{
    DynAccelerationStructure, DynBindGroup, DynBindGroupLayout, DynCommandEncoder,
    DynPipelineLayout, DynQueue, DynResourceExt as _, DynSampler, DynShaderModule, DynTexture,
    DynTextureView,
};

pub trait DynDevice: DynResource {
    unsafe fn create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> Result<Box<dyn DynBuffer>, DeviceError>;

    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>);

    unsafe fn map_buffer(
        &self,
        buffer: &dyn DynBuffer,
        range: MemoryRange,
    ) -> Result<BufferMapping, DeviceError>;

    unsafe fn unmap_buffer(&self, buffer: &dyn DynBuffer);

    unsafe fn flush_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]);
    unsafe fn invalidate_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]);

    unsafe fn create_texture(
        &self,
        desc: &TextureDescriptor,
    ) -> Result<Box<dyn DynTexture>, DeviceError>;
    unsafe fn destroy_texture(&self, texture: Box<dyn DynTexture>);
    unsafe fn create_texture_view(
        &self,
        texture: &dyn DynTexture,
        desc: &TextureViewDescriptor,
    ) -> Result<Box<dyn DynTextureView>, DeviceError>;
    unsafe fn destroy_texture_view(&self, view: Box<dyn DynTextureView>);
    unsafe fn create_sampler(
        &self,
        desc: &SamplerDescriptor,
    ) -> Result<Box<dyn DynSampler>, DeviceError>;
    unsafe fn destroy_sampler(&self, sampler: Box<dyn DynSampler>);

    unsafe fn create_command_encoder(
        &self,
        desc: &CommandEncoderDescriptor<dyn DynQueue>,
    ) -> Result<Box<dyn DynCommandEncoder>, DeviceError>;
    unsafe fn destroy_command_encoder(&self, pool: Box<dyn DynCommandEncoder>);

    unsafe fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<Box<dyn DynBindGroupLayout>, DeviceError>;
    unsafe fn destroy_bind_group_layout(&self, bg_layout: Box<dyn DynBindGroupLayout>);

    unsafe fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDescriptor<dyn DynBindGroupLayout>,
    ) -> Result<Box<dyn DynPipelineLayout>, DeviceError>;
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: Box<dyn DynPipelineLayout>);

    unsafe fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<
            dyn DynBindGroupLayout,
            dyn DynBuffer,
            dyn DynSampler,
            dyn DynTextureView,
            dyn DynAccelerationStructure,
        >,
    ) -> Result<Box<dyn DynBindGroup>, DeviceError>;
    unsafe fn destroy_bind_group(&self, group: Box<dyn DynBindGroup>);

    unsafe fn create_shader_module(
        &self,
        desc: &ShaderModuleDescriptor,
        shader: ShaderInput,
    ) -> Result<Box<dyn DynShaderModule>, ShaderError>;
    unsafe fn destroy_shader_module(&self, module: Box<dyn DynShaderModule>);
}

impl<D: Device + DynResource> DynDevice for D {
    unsafe fn create_buffer(
        &self,
        desc: &BufferDescriptor,
    ) -> Result<Box<dyn DynBuffer>, DeviceError> {
        unsafe { D::create_buffer(self, desc) }.map(|b| -> Box<dyn DynBuffer> { Box::new(b) })
    }

    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>) {
        unsafe { D::destroy_buffer(self, buffer.unbox()) };
    }

    unsafe fn map_buffer(
        &self,
        buffer: &dyn DynBuffer,
        range: MemoryRange,
    ) -> Result<BufferMapping, DeviceError> {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::map_buffer(self, buffer, range) }
    }

    unsafe fn unmap_buffer(&self, buffer: &dyn DynBuffer) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::unmap_buffer(self, buffer) }
    }

    unsafe fn flush_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::flush_mapped_ranges(self, buffer, ranges.iter().cloned()) }
    }

    unsafe fn invalidate_mapped_ranges(&self, buffer: &dyn DynBuffer, ranges: &[MemoryRange]) {
        let buffer = buffer.expect_downcast_ref();
        unsafe { D::invalidate_mapped_ranges(self, buffer, ranges.iter().cloned()) }
    }

    unsafe fn create_texture(
        &self,
        desc: &TextureDescriptor,
    ) -> Result<Box<dyn DynTexture>, DeviceError> {
        unsafe { D::create_texture(self, desc) }.map(|b| {
            let boxed_texture: Box<<D::A as Api>::Texture> = Box::new(b);
            let boxed_texture: Box<dyn DynTexture> = boxed_texture;
            boxed_texture
        })
    }

    unsafe fn destroy_texture(&self, texture: Box<dyn DynTexture>) {
        unsafe { D::destroy_texture(self, texture.unbox()) };
    }

    unsafe fn create_texture_view(
        &self,
        texture: &dyn DynTexture,
        desc: &TextureViewDescriptor,
    ) -> Result<Box<dyn DynTextureView>, DeviceError> {
        let texture = texture.expect_downcast_ref();
        unsafe { D::create_texture_view(self, texture, desc) }.map(|b| {
            let boxed_texture_view: Box<<D::A as Api>::TextureView> = Box::new(b);
            let boxed_texture_view: Box<dyn DynTextureView> = boxed_texture_view;
            boxed_texture_view
        })
    }

    unsafe fn destroy_texture_view(&self, view: Box<dyn DynTextureView>) {
        unsafe { D::destroy_texture_view(self, view.unbox()) };
    }

    unsafe fn create_sampler(
        &self,
        desc: &SamplerDescriptor,
    ) -> Result<Box<dyn DynSampler>, DeviceError> {
        unsafe { D::create_sampler(self, desc) }.map(|b| {
            let boxed_sampler: Box<<D::A as Api>::Sampler> = Box::new(b);
            let boxed_sampler: Box<dyn DynSampler> = boxed_sampler;
            boxed_sampler
        })
    }

    unsafe fn destroy_sampler(&self, sampler: Box<dyn DynSampler>) {
        unsafe { D::destroy_sampler(self, sampler.unbox()) };
    }

    unsafe fn create_command_encoder(
        &self,
        desc: &CommandEncoderDescriptor<'_, dyn DynQueue>,
    ) -> Result<Box<dyn DynCommandEncoder>, DeviceError> {
        let desc = CommandEncoderDescriptor {
            label: desc.label,
            queue: desc.queue.expect_downcast_ref(),
        };
        unsafe { D::create_command_encoder(self, &desc) }
            .map(|b| Box::new(b) as Box<dyn DynCommandEncoder>)
    }

    unsafe fn destroy_command_encoder(&self, encoder: Box<dyn DynCommandEncoder>) {
        unsafe { D::destroy_command_encoder(self, encoder.unbox()) };
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &BindGroupLayoutDescriptor,
    ) -> Result<Box<dyn DynBindGroupLayout>, DeviceError> {
        unsafe { D::create_bind_group_layout(self, desc) }
            .map(|b| Box::new(b) as Box<dyn DynBindGroupLayout>)
    }

    unsafe fn destroy_bind_group_layout(&self, bg_layout: Box<dyn DynBindGroupLayout>) {
        unsafe { D::destroy_bind_group_layout(self, bg_layout.unbox()) };
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &PipelineLayoutDescriptor<dyn DynBindGroupLayout>,
    ) -> Result<Box<dyn DynPipelineLayout>, DeviceError> {
        let bind_group_layouts: Vec<_> = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.expect_downcast_ref())
            .collect();
        let desc = PipelineLayoutDescriptor {
            label: desc.label,
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: desc.push_constant_ranges,
            flags: desc.flags,
        };

        unsafe { D::create_pipeline_layout(self, &desc) }
            .map(|b| Box::new(b) as Box<dyn DynPipelineLayout>)
    }

    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: Box<dyn DynPipelineLayout>) {
        unsafe { D::destroy_pipeline_layout(self, pipeline_layout.unbox()) };
    }

    unsafe fn create_bind_group(
        &self,
        desc: &BindGroupDescriptor<
            dyn DynBindGroupLayout,
            dyn DynBuffer,
            dyn DynSampler,
            dyn DynTextureView,
            dyn DynAccelerationStructure,
        >,
    ) -> Result<Box<dyn DynBindGroup>, DeviceError> {
        let buffers: Vec<_> = desc
            .buffers
            .iter()
            .map(|b| b.clone().expect_downcast())
            .collect();
        let samplers: Vec<_> = desc
            .samplers
            .iter()
            .map(|s| s.expect_downcast_ref())
            .collect();
        let textures: Vec<_> = desc
            .textures
            .iter()
            .map(|t| t.clone().expect_downcast())
            .collect();
        let acceleration_structures: Vec<_> = desc
            .acceleration_structures
            .iter()
            .map(|a| a.expect_downcast_ref())
            .collect();

        let desc = BindGroupDescriptor {
            label: desc.label.to_owned(),
            layout: desc.layout.expect_downcast_ref(),
            buffers: &buffers,
            samplers: &samplers,
            textures: &textures,
            entries: desc.entries,
            acceleration_structures: &acceleration_structures,
        };

        unsafe { D::create_bind_group(self, &desc) }.map(|b| Box::new(b) as Box<dyn DynBindGroup>)
    }

    unsafe fn destroy_bind_group(&self, group: Box<dyn DynBindGroup>) {
        unsafe { D::destroy_bind_group(self, group.unbox()) };
    }

    unsafe fn create_shader_module(
        &self,
        desc: &ShaderModuleDescriptor,
        shader: ShaderInput,
    ) -> Result<Box<dyn DynShaderModule>, ShaderError> {
        unsafe { D::create_shader_module(self, desc, shader) }
            .map(|b| Box::new(b) as Box<dyn DynShaderModule>)
    }

    unsafe fn destroy_shader_module(&self, module: Box<dyn DynShaderModule>) {
        unsafe { D::destroy_shader_module(self, module.unbox()) };
    }
}
