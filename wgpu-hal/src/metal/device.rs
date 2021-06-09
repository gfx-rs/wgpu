use super::conv;
use std::ptr;

type DeviceResult<T> = Result<T, crate::DeviceError>;

impl crate::Device<super::Api> for super::Device {
    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<super::Buffer> {
        let map_read = desc.usage.contains(crate::BufferUse::MAP_READ);
        let map_write = desc.usage.contains(crate::BufferUse::MAP_WRITE);

        let mut options = mtl::MTLResourceOptions::empty();
        options |= if map_read || map_write {
            mtl::MTLResourceOptions::StorageModeShared
        } else {
            mtl::MTLResourceOptions::StorageModePrivate
        };
        options.set(mtl::MTLResourceOptions::CPUCacheModeDefaultCache, map_read);

        //TODO: HazardTrackingModeUntracked

        let raw = self.shared.device.lock().new_buffer(desc.size, options);
        if let Some(label) = desc.label {
            raw.set_label(label);
        }
        Ok(super::Buffer {
            raw,
            size: desc.size,
            options,
        })
    }
    unsafe fn destroy_buffer(&self, _buffer: super::Buffer) {}

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> DeviceResult<ptr::NonNull<u8>> {
        let ptr = buffer.raw.contents() as *mut u8;
        assert!(!ptr.is_null());
        Ok(ptr::NonNull::new(ptr.offset(range.start as isize)).unwrap())
    }

    unsafe fn unmap_buffer(&self, _buffer: &super::Buffer) -> DeviceResult<()> {
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> DeviceResult<super::Texture> {
        let mtl_format = self.shared.private_caps.map_format(desc.format);

        let descriptor = mtl::TextureDescriptor::new();
        let mut array_layers = desc.size.depth_or_array_layers;
        let mtl_type = match desc.dimension {
            wgt::TextureDimension::D1 => {
                if desc.size.depth_or_array_layers > 1 {
                    descriptor.set_array_length(desc.size.depth_or_array_layers as u64);
                    mtl::MTLTextureType::D1Array
                } else {
                    mtl::MTLTextureType::D1
                }
            }
            wgt::TextureDimension::D2 => {
                if desc.sample_count > 1 {
                    descriptor.set_sample_count(desc.sample_count as u64);
                    mtl::MTLTextureType::D2Multisample
                } else if desc.size.depth_or_array_layers > 1 {
                    descriptor.set_array_length(desc.size.depth_or_array_layers as u64);
                    mtl::MTLTextureType::D2Array
                } else {
                    mtl::MTLTextureType::D2
                }
            }
            wgt::TextureDimension::D3 => {
                descriptor.set_depth(desc.size.depth_or_array_layers as u64);
                array_layers = 1;
                mtl::MTLTextureType::D3
            }
        };

        descriptor.set_texture_type(mtl_type);
        descriptor.set_width(desc.size.width as u64);
        descriptor.set_height(desc.size.height as u64);
        descriptor.set_mipmap_level_count(desc.mip_level_count as u64);
        descriptor.set_pixel_format(mtl_format);
        descriptor.set_usage(conv::map_texture_usage(desc.usage));
        descriptor.set_storage_mode(mtl::MTLStorageMode::Private);

        let raw = self.shared.device.lock().new_texture(&descriptor);
        if let Some(label) = desc.label {
            raw.set_label(label);
        }

        Ok(super::Texture {
            raw,
            raw_format: mtl_format,
            raw_type: mtl_type,
            mip_levels: desc.mip_level_count,
            array_layers,
        })
    }

    unsafe fn destroy_texture(&self, _texture: super::Texture) {}

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<super::TextureView> {
        let mtl_format = self.shared.private_caps.map_format(desc.format);

        let mtl_type = if texture.raw_type == mtl::MTLTextureType::D2Multisample {
            texture.raw_type
        } else {
            conv::map_texture_view_dimension(desc.dimension)
        };

        let raw = if mtl_format == texture.raw_format
            && mtl_type == texture.raw_type
            && desc.range == wgt::ImageSubresourceRange::default()
        {
            // Some images are marked as framebuffer-only, and we can't create aliases of them.
            // Also helps working around Metal bugs with aliased array textures.
            texture.raw.to_owned()
        } else {
            let mip_level_count = match desc.range.mip_level_count {
                Some(count) => count.get(),
                None => texture.mip_levels - desc.range.base_mip_level,
            };
            let array_layer_count = match desc.range.array_layer_count {
                Some(count) => count.get(),
                None => texture.array_layers - desc.range.base_array_layer,
            };

            texture.raw.new_texture_view_from_slice(
                mtl_format,
                mtl_type,
                mtl::NSRange {
                    location: desc.range.base_mip_level as _,
                    length: mip_level_count as _,
                },
                mtl::NSRange {
                    location: desc.range.base_array_layer as _,
                    length: array_layer_count as _,
                },
            )
        };

        Ok(super::TextureView { raw })
    }
    unsafe fn destroy_texture_view(&self, _view: super::TextureView) {}

    unsafe fn create_sampler(&self, desc: &crate::SamplerDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, sampler: Resource) {}

    unsafe fn create_command_buffer(
        &self,
        desc: &crate::CommandBufferDescriptor,
    ) -> DeviceResult<Encoder> {
        Ok(Encoder)
    }
    unsafe fn destroy_command_buffer(&self, cmd_buf: Encoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group_layout(&self, bg_layout: Resource) {}
    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: Resource) {}
    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<Api>,
    ) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_bind_group(&self, group: Resource) {}

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::NagaShader,
    ) -> Result<Resource, (crate::ShaderError, crate::NagaShader)> {
        Ok(Resource)
    }
    unsafe fn destroy_shader_module(&self, module: Resource) {}
    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: Resource) {}
    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<Api>,
    ) -> Result<Resource, crate::PipelineError> {
        Ok(Resource)
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: Resource) {}

    unsafe fn create_query_set(&self, desc: &wgt::QuerySetDescriptor) -> DeviceResult<Resource> {
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
