//TODO: remove this
use super::{Encoder, Resource};
type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self) {}
    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_buffer(&self, buffer: Resource) {}
    unsafe fn map_buffer(
        &self,
        buffer: &Resource,
        range: crate::MemoryRange,
    ) -> DeviceResult<crate::BufferMapping> {
        Err(crate::DeviceError::Lost)
    }
    unsafe fn unmap_buffer(&self, buffer: &Resource) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &Resource, ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &Resource, ranges: I) {}

    unsafe fn create_texture(&self, desc: &crate::TextureDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_texture(&self, texture: Resource) {}
    unsafe fn create_texture_view(
        &self,
        texture: &Resource,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_texture_view(&self, view: Resource) {}
    unsafe fn create_sampler(&self, desc: &crate::SamplerDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, sampler: Resource) {}

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> DeviceResult<Encoder> {
        Ok(Encoder)
    }
    unsafe fn destroy_command_encoder(&self, encoder: Encoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group_layout(&self, bg_layout: Resource) {}
    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: Resource) {}
    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group(&self, group: Resource) {}

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<Resource, crate::ShaderError> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: Resource) {}

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_query_set(&self, set: Resource) {}
    unsafe fn create_fence(&self) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_fence(&self, fence: Resource) {}
    unsafe fn get_fence_value(&self, fence: &Resource) -> DeviceResult<crate::FenceValue> {
        Ok(0)
    }
    unsafe fn wait(
        &self,
        fence: &Resource,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> DeviceResult<bool> {
        Ok(true)
    }

    unsafe fn start_capture(&self) -> bool {
        false
    }
    unsafe fn stop_capture(&self) {}
}
