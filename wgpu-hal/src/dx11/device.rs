use std::{ffi::c_void, mem};

use winapi::um::d3d11;

use crate::auxil::dxgi::result::HResult;

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self, queue: super::Queue) {
        todo!()
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        todo!()
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        todo!()
    }

    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        todo!()
    }

    unsafe fn flush_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        todo!()
    }

    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        todo!()
    }

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_texture(&self, texture: super::Texture) {
        todo!()
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_texture_view(&self, view: super::TextureView) {
        todo!()
    }

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        todo!()
    }

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_command_encoder(&self, pool: super::CommandEncoder) {
        todo!()
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<super::BindGroupLayout, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_bind_group_layout(&self, bg_layout: super::BindGroupLayout) {
        todo!()
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> Result<super::PipelineLayout, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: super::PipelineLayout) {
        todo!()
    }

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> Result<super::BindGroup, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_bind_group(&self, group: super::BindGroup) {
        todo!()
    }

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        todo!()
    }

    unsafe fn destroy_shader_module(&self, module: super::ShaderModule) {
        todo!()
    }

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        todo!()
    }

    unsafe fn destroy_render_pipeline(&self, pipeline: super::RenderPipeline) {
        todo!()
    }

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        todo!()
    }

    unsafe fn destroy_compute_pipeline(&self, pipeline: super::ComputePipeline) {
        todo!()
    }

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<super::QuerySet, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_query_set(&self, set: super::QuerySet) {
        todo!()
    }

    unsafe fn create_fence(&self) -> Result<super::Fence, crate::DeviceError> {
        todo!()
    }

    unsafe fn destroy_fence(&self, fence: super::Fence) {
        todo!()
    }

    unsafe fn get_fence_value(
        &self,
        fence: &super::Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        todo!()
    }

    unsafe fn wait(
        &self,
        fence: &super::Fence,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        todo!()
    }

    unsafe fn start_capture(&self) -> bool {
        todo!()
    }

    unsafe fn stop_capture(&self) {
        todo!()
    }
}

impl crate::Queue<super::Api> for super::Queue {
    unsafe fn submit(
        &mut self,
        command_buffers: &[&super::CommandBuffer],
        signal_fence: Option<(&mut super::Fence, crate::FenceValue)>,
    ) -> Result<(), crate::DeviceError> {
        todo!()
    }

    unsafe fn present(
        &mut self,
        surface: &mut super::Surface,
        texture: super::SurfaceTexture,
    ) -> Result<(), crate::SurfaceError> {
        todo!()
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        todo!()
    }
}

impl super::D3D11Device {
    #[allow(trivial_casts)] // come on
    pub unsafe fn check_feature_support<T>(&self, feature: d3d11::D3D11_FEATURE) -> T {
        unsafe {
            let mut value = mem::zeroed::<T>();
            let ret = self.CheckFeatureSupport(
                feature,
                &mut value as *mut T as *mut c_void,
                mem::size_of::<T>() as u32,
            );
            assert_eq!(ret.into_result(), Ok(()));

            value
        }
    }
}
