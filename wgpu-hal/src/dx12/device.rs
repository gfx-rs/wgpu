use super::{conv, descriptor, HResult as _};
use parking_lot::Mutex;
use std::{iter, mem, ptr};
use winapi::{
    shared::{dxgiformat, dxgitype},
    um::{d3d12, d3d12sdklayers, synchapi, winbase},
    Interface,
};

//TODO: remove this
use super::Resource;

type DeviceResult<T> = Result<T, crate::DeviceError>;

const D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING: u32 = 0x1688;

fn wide_cstr(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(iter::once(0)).collect()
}

impl super::Device {
    pub(super) fn new(
        raw: native::Device,
        present_queue: native::CommandQueue,
        private_caps: super::PrivateCapabilities,
    ) -> Result<Self, crate::DeviceError> {
        let mut idle_fence = native::Fence::null();
        let hr = unsafe {
            raw.CreateFence(
                0,
                d3d12::D3D12_FENCE_FLAG_NONE,
                &d3d12::ID3D12Fence::uuidof(),
                idle_fence.mut_void(),
            )
        };
        hr.to_device_result("Idle fence creation")?;

        Ok(super::Device {
            raw,
            present_queue,
            idler: super::Idler {
                fence: idle_fence,
                event: native::Event::create(false, false),
            },
            private_caps,
            rtv_pool: Mutex::new(descriptor::CpuPool::new(
                raw,
                native::DescriptorHeapType::Rtv,
            )),
            dsv_pool: Mutex::new(descriptor::CpuPool::new(
                raw,
                native::DescriptorHeapType::Dsv,
            )),
            srv_uav_pool: Mutex::new(descriptor::CpuPool::new(
                raw,
                native::DescriptorHeapType::CbvSrvUav,
            )),
        })
    }

    pub(super) unsafe fn wait_idle(&self) -> Result<(), crate::DeviceError> {
        let value = self.idler.fence.get_value() + 1;
        log::info!("Waiting for idle with value {}", value);
        self.present_queue.signal(self.idler.fence, value);
        let hr = self
            .idler
            .fence
            .set_event_on_completion(self.idler.event, value);
        hr.to_device_result("Set event")?;
        synchapi::WaitForSingleObject(self.idler.event.0, winbase::INFINITE);
        Ok(())
    }

    unsafe fn view_texture_as_shader_resource(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> descriptor::Handle {
        let mut raw_desc = d3d12::D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: conv::map_texture_format(desc.format),
            ViewDimension: 0,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            u: mem::zeroed(),
        };

        #[allow(non_snake_case)]
        let MipLevels = match desc.range.mip_level_count {
            Some(count) => count.get(),
            None => !0,
        };
        #[allow(non_snake_case)]
        let ArraySize = match desc.range.array_layer_count {
            Some(count) => count.get(),
            None => texture.size.depth_or_array_layers - desc.range.base_array_layer,
        };

        match desc.dimension {
            wgt::TextureViewDimension::D1 => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE1D;
                *raw_desc.u.Texture1D_mut() = d3d12::D3D12_TEX1D_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    ResourceMinLODClamp: 0.0,
                }
            }
            /*
            wgt::TextureViewDimension::D1Array => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
                *raw_desc.u.Texture1DArray_mut() = d3d12::D3D12_TEX1D_ARRAY_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize,
                    ResourceMinLODClamp: 0.0,
                }
            }*/
            wgt::TextureViewDimension::D2 if texture.sample_count > 1 => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE2DMS;
                *raw_desc.u.Texture2DMS_mut() = d3d12::D3D12_TEX2DMS_SRV {
                    UnusedField_NothingToDefine: 0,
                }
            }
            wgt::TextureViewDimension::D2 => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE2D;
                *raw_desc.u.Texture2D_mut() = d3d12::D3D12_TEX2D_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                }
            }
            wgt::TextureViewDimension::D2Array if texture.sample_count > 1 => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
                *raw_desc.u.Texture2DMSArray_mut() = d3d12::D3D12_TEX2DMS_ARRAY_SRV {
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize,
                }
            }
            wgt::TextureViewDimension::D2Array => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                *raw_desc.u.Texture2DArray_mut() = d3d12::D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                }
            }
            wgt::TextureViewDimension::D3 => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE3D;
                *raw_desc.u.Texture3D_mut() = d3d12::D3D12_TEX3D_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    ResourceMinLODClamp: 0.0,
                }
            }
            wgt::TextureViewDimension::Cube => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURECUBE;
                *raw_desc.u.TextureCube_mut() = d3d12::D3D12_TEXCUBE_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    ResourceMinLODClamp: 0.0,
                }
            }
            wgt::TextureViewDimension::CubeArray => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
                *raw_desc.u.TextureCubeArray_mut() = d3d12::D3D12_TEXCUBE_ARRAY_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    First2DArrayFace: desc.range.base_array_layer,
                    NumCubes: ArraySize / 6,
                    ResourceMinLODClamp: 0.0,
                }
            }
        }

        let handle = self.srv_uav_pool.lock().alloc_handle();
        self.raw
            .CreateShaderResourceView(texture.resource.as_mut_ptr(), &raw_desc, handle.raw);
        handle
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self) {
        //self.heap_srv_cbv_uav.0.destroy();
        //self.samplers.destroy();
        self.rtv_pool.into_inner().destroy();
        self.dsv_pool.into_inner().destroy();
        self.srv_uav_pool.into_inner().destroy();

        //self.descriptor_updater.lock().destroy();

        // Debug tracking alive objects
        if let Ok(debug_device) = self
            .raw
            .cast::<d3d12sdklayers::ID3D12DebugDevice>()
            .to_result()
        {
            debug_device.ReportLiveDeviceObjects(d3d12sdklayers::D3D12_RLDO_DETAIL);
            debug_device.destroy();
        }

        self.raw.destroy();
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let mut resource = native::Resource::null();

        let raw_desc = d3d12::D3D12_RESOURCE_DESC {
            Dimension: d3d12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: desc.size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: dxgiformat::DXGI_FORMAT_UNKNOWN,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: d3d12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: conv::map_buffer_usage_to_resource_flags(desc.usage),
        };

        let is_cpu_read = desc.usage.contains(crate::BufferUses::MAP_READ);
        let is_cpu_write = desc.usage.contains(crate::BufferUses::MAP_WRITE);

        let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
            Type: d3d12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: if is_cpu_read {
                d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
            } else if is_cpu_write {
                d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE
            } else {
                d3d12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE
            },
            MemoryPoolPreference: match self.private_caps.memory_architecture {
                super::MemoryArchitecture::NonUnified if !is_cpu_read && !is_cpu_write => {
                    d3d12::D3D12_MEMORY_POOL_L1
                }
                _ => d3d12::D3D12_MEMORY_POOL_L0,
            },
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let hr = self.raw.CreateCommittedResource(
            &heap_properties,
            d3d12::D3D12_HEAP_FLAG_NONE,
            &raw_desc,
            d3d12::D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &d3d12::ID3D12Resource::uuidof(),
            resource.mut_void(),
        );

        hr.to_device_result("Buffer creation")?;
        Ok(super::Buffer { resource })
    }
    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        buffer.resource.destroy();
    }
    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let mut ptr = ptr::null_mut();
        let hr = (*buffer.resource).Map(0, &d3d12::D3D12_RANGE { Begin: 0, End: 0 }, &mut ptr);
        hr.to_device_result("Map buffer")?;
        Ok(crate::BufferMapping {
            ptr: ptr::NonNull::new(ptr.offset(range.start as isize) as *mut _).unwrap(),
            //TODO: double-check this. Documentation is a bit misleading -
            // it implies that Map/Unmap is needed to invalidate/flush memory.
            is_coherent: true,
        })
    }
    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        (*buffer.resource).Unmap(0, &d3d12::D3D12_RANGE { Begin: 0, End: 0 });
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, _buffer: &super::Buffer, ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, ranges: I) {}

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let mut resource = native::Resource::null();

        let raw_desc = d3d12::D3D12_RESOURCE_DESC {
            Dimension: conv::map_texture_dimension(desc.dimension),
            Alignment: 0,
            Width: desc.size.width as u64,
            Height: desc.size.height,
            DepthOrArraySize: desc.size.depth_or_array_layers as u16,
            MipLevels: desc.mip_level_count as u16,
            //TODO: map to surface format to allow view casting
            Format: conv::map_texture_format(desc.format),
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: desc.sample_count,
                Quality: 0,
            },
            Layout: d3d12::D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            Flags: conv::map_texture_usage_to_resource_flags(desc.usage),
        };

        let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
            Type: d3d12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: d3d12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: match self.private_caps.memory_architecture {
                super::MemoryArchitecture::NonUnified => d3d12::D3D12_MEMORY_POOL_L1,
                super::MemoryArchitecture::Unified { .. } => d3d12::D3D12_MEMORY_POOL_L0,
            },
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let hr = self.raw.CreateCommittedResource(
            &heap_properties,
            d3d12::D3D12_HEAP_FLAG_NONE,
            &raw_desc,
            d3d12::D3D12_RESOURCE_STATE_COMMON,
            ptr::null(),
            &d3d12::ID3D12Resource::uuidof(),
            resource.mut_void(),
        );

        if let Some(label) = desc.label {
            let cwstr = wide_cstr(label);
            resource.SetName(cwstr.as_ptr());
        }

        hr.to_device_result("Texture creation")?;
        Ok(super::Texture {
            resource,
            size: desc.size,
            sample_count: desc.sample_count,
        })
    }
    unsafe fn destroy_texture(&self, texture: super::Texture) {
        texture.resource.destroy();
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        Ok(super::TextureView {
            handle_srv: if desc
                .usage
                .intersects(crate::TextureUses::SAMPLED | crate::TextureUses::STORAGE_LOAD)
            {
                Some(self.view_texture_as_shader_resource(texture, desc))
            } else {
                None
            },
            handle_rtv: if desc.usage.intersects(crate::TextureUses::COLOR_TARGET) {
                unimplemented!()
            } else {
                None
            },
            handle_dsv: if desc.usage.intersects(
                crate::TextureUses::DEPTH_STENCIL_READ | crate::TextureUses::DEPTH_STENCIL_WRITE,
            ) {
                unimplemented!()
            } else {
                None
            },
            handle_uav: if desc.usage.intersects(crate::TextureUses::STORAGE_STORE) {
                unimplemented!()
            } else {
                None
            },
        })
    }
    unsafe fn destroy_texture_view(&self, view: super::TextureView) {
        if let Some(handle) = view.handle_srv {
            self.srv_uav_pool.lock().free_handle(handle);
        }
        if let Some(handle) = view.handle_uav {
            self.srv_uav_pool.lock().free_handle(handle);
        }
        if let Some(handle) = view.handle_rtv {
            self.rtv_pool.lock().free_handle(handle);
        }
        if let Some(handle) = view.handle_dsv {
            self.dsv_pool.lock().free_handle(handle);
        }
    }

    unsafe fn create_sampler(&self, desc: &crate::SamplerDescriptor) -> DeviceResult<Resource> {
        Ok(Resource)
    }
    unsafe fn destroy_sampler(&self, sampler: Resource) {}

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        Ok(super::CommandEncoder {})
    }
    unsafe fn destroy_command_encoder(&self, encoder: super::CommandEncoder) {}

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
