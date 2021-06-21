use super::{DeviceResult, Encoder, Resource}; //TEMP
use glow::HasContext;
use std::{convert::TryInto, ptr::NonNull};

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self) {
        let gl = &self.shared.context;
        gl.delete_vertex_array(self.main_vao);
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let gl = &self.shared.context;

        let target = if desc.usage.contains(crate::BufferUse::INDEX) {
            glow::ELEMENT_ARRAY_BUFFER
        } else {
            glow::ARRAY_BUFFER
        };
        let map_coherent = false;
        let map_flags = glow::MAP_PERSISTENT_BIT
            | if map_coherent {
                glow::MAP_COHERENT_BIT
            } else {
                0
            };
        let mut storage_flags = 0;
        if desc.usage.contains(crate::BufferUse::MAP_READ) {
            storage_flags |= map_flags | glow::MAP_READ_BIT;
        }
        if desc.usage.contains(crate::BufferUse::MAP_WRITE) {
            storage_flags |= map_flags | glow::MAP_WRITE_BIT;
        }

        let raw = gl.create_buffer().unwrap();
        gl.bind_buffer(target, Some(raw));
        let raw_size = desc
            .size
            .try_into()
            .map_err(|_| crate::DeviceError::OutOfMemory)?;
        gl.buffer_storage(target, raw_size, None, storage_flags);
        gl.bind_buffer(target, None);

        Ok(super::Buffer {
            raw,
            target,
            map_flags,
        })
    }
    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        let gl = &self.shared.context;
        gl.delete_buffer(buffer.raw);
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let gl = &self.shared.context;

        gl.bind_buffer(buffer.target, Some(buffer.raw));
        let ptr = gl.map_buffer_range(
            buffer.target,
            range.start as i32,
            (range.end - range.start) as i32,
            buffer.map_flags,
        );
        gl.bind_buffer(buffer.target, None);

        Ok(crate::BufferMapping {
            ptr: NonNull::new(ptr).ok_or(crate::DeviceError::Lost)?,
            is_coherent: buffer.map_flags & glow::MAP_COHERENT_BIT != 0,
        })
    }
    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> DeviceResult<()> {
        let gl = &self.shared.context;
        gl.bind_buffer(buffer.target, Some(buffer.raw));
        gl.unmap_buffer(buffer.target);
        gl.bind_buffer(buffer.target, None);
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let gl = &self.shared.context;
        for range in ranges {
            gl.flush_mapped_buffer_range(
                buffer.target,
                range.start as i32,
                (range.end - range.start) as i32,
            );
        }
    }
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let gl = &self.shared.context;
        for range in ranges {
            gl.invalidate_buffer_sub_data(
                buffer.target,
                range.start as i32,
                (range.end - range.start) as i32,
            );
        }
    }

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let gl = &self.shared.context;

        let render_usage = crate::TextureUse::COLOR_TARGET
            | crate::TextureUse::DEPTH_STENCIL_WRITE
            | crate::TextureUse::DEPTH_STENCIL_READ;
        let format_desc = self.shared.describe_texture_format(desc.format);
        Ok(
            if render_usage.contains(desc.usage)
                && desc.dimension == wgt::TextureDimension::D2
                && desc.size.depth_or_array_layers == 1
            {
                let raw = gl.create_renderbuffer().unwrap();
                gl.bind_renderbuffer(glow::RENDERBUFFER, Some(raw));
                if desc.sample_count > 1 {
                    gl.renderbuffer_storage_multisample(
                        glow::RENDERBUFFER,
                        desc.sample_count as i32,
                        format_desc.tex_internal,
                        desc.size.width as i32,
                        desc.size.height as i32,
                    );
                } else {
                    gl.renderbuffer_storage(
                        glow::RENDERBUFFER,
                        format_desc.tex_internal,
                        desc.size.width as i32,
                        desc.size.height as i32,
                    );
                }
                super::Texture::Renderbuffer {
                    raw,
                    aspects: desc.format.into(),
                }
            } else {
                let raw = gl.create_texture().unwrap();
                let target = match desc.dimension {
                    wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => {
                        if desc.sample_count > 1 {
                            let target = glow::TEXTURE_2D;
                            gl.bind_texture(target, Some(raw));
                            // https://github.com/grovesNL/glow/issues/169
                            //gl.tex_storage_2d_multisample(target, desc.sample_count as i32, format_desc.tex_internal, desc.size.width as i32, desc.size.height as i32, true);
                            log::error!("TODO: support `tex_storage_2d_multisample` (https://github.com/grovesNL/glow/issues/169)");
                            return Err(crate::DeviceError::Lost);
                        } else if desc.size.depth_or_array_layers > 1 {
                            let target = glow::TEXTURE_2D_ARRAY;
                            gl.bind_texture(target, Some(raw));
                            gl.tex_storage_3d(
                                target,
                                desc.mip_level_count as i32,
                                format_desc.tex_internal,
                                desc.size.width as i32,
                                desc.size.height as i32,
                                desc.size.depth_or_array_layers as i32,
                            );
                            target
                        } else {
                            let target = glow::TEXTURE_2D;
                            gl.bind_texture(target, Some(raw));
                            gl.tex_storage_2d(
                                target,
                                desc.mip_level_count as i32,
                                format_desc.tex_internal,
                                desc.size.width as i32,
                                desc.size.height as i32,
                            );
                            target
                        }
                    }
                    wgt::TextureDimension::D3 => {
                        let target = glow::TEXTURE_3D;
                        gl.bind_texture(target, Some(raw));
                        gl.tex_storage_3d(
                            target,
                            desc.mip_level_count as i32,
                            format_desc.tex_internal,
                            desc.size.width as i32,
                            desc.size.height as i32,
                            desc.size.depth_or_array_layers as i32,
                        );
                        target
                    }
                };
                super::Texture::Texture { raw, target }
            },
        )
    }
    unsafe fn destroy_texture(&self, texture: super::Texture) {
        let gl = &self.shared.context;
        match texture {
            super::Texture::Renderbuffer { raw, .. } => {
                gl.delete_renderbuffer(raw);
            }
            super::Texture::Texture { raw, target } => {
                gl.delete_texture(raw);
            }
        }
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        Ok(match *texture {
            super::Texture::Renderbuffer { raw, aspects } => super::TextureView::Renderbuffer {
                raw,
                aspects: aspects & crate::FormatAspect::from(desc.range.aspect),
            },
            super::Texture::Texture { raw, target } => super::TextureView::Texture {
                raw,
                target,
                range: desc.range.clone(),
            },
        })
    }
    unsafe fn destroy_texture_view(&self, view: super::TextureView) {}

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        use super::Sampled;
        let gl = &self.shared.context;

        let raw = gl.create_sampler().unwrap();
        super::SamplerBinding(raw).configure_sampling(gl, desc);

        Ok(super::Sampler { raw })
    }
    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        let gl = &self.shared.context;
        gl.delete_sampler(sampler.raw);
    }

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
