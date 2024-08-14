// Box casts are needed, alternative would be a temporaries which are more verbose and not more expressive.
#![allow(trivial_casts)]

use crate::{
    Api, BufferDescriptor, BufferMapping, CommandEncoderDescriptor, Device, DeviceError, DynBuffer,
    DynResource, MemoryRange, SamplerDescriptor, TextureDescriptor, TextureViewDescriptor,
};

use super::{
    DynCommandEncoder, DynQueue, DynResourceExt as _, DynSampler, DynTexture, DynTextureView,
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
}
