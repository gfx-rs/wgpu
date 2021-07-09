use super::{conv, descriptor, HResult as _};
use parking_lot::Mutex;
use std::{mem, ptr, slice, sync::Arc};
use winapi::{
    shared::{dxgiformat, dxgitype, winerror},
    um::{d3d12, d3d12sdklayers, synchapi, winbase},
    Interface,
};

//TODO: remove this
use super::Resource;

const D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING: u32 = 0x1688;

impl super::Device {
    pub(super) fn new(
        raw: native::Device,
        present_queue: native::CommandQueue,
        private_caps: super::PrivateCapabilities,
        library: &Arc<native::D3D12Lib>,
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
        hr.into_device_result("Idle fence creation")?;

        let mut zero_buffer = native::Resource::null();
        unsafe {
            let raw_desc = d3d12::D3D12_RESOURCE_DESC {
                Dimension: d3d12::D3D12_RESOURCE_DIMENSION_BUFFER,
                Alignment: 0,
                Width: super::ZERO_BUFFER_SIZE,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: dxgiformat::DXGI_FORMAT_UNKNOWN,
                SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: d3d12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: d3d12::D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
            };

            let heap_properties = d3d12::D3D12_HEAP_PROPERTIES {
                Type: d3d12::D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                MemoryPoolPreference: d3d12::D3D12_MEMORY_POOL_L0,
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            };

            let hr = raw.CreateCommittedResource(
                &heap_properties,
                d3d12::D3D12_HEAP_FLAG_NONE,
                &raw_desc,
                d3d12::D3D12_RESOURCE_STATE_COMMON,
                ptr::null(),
                &d3d12::ID3D12Resource::uuidof(),
                zero_buffer.mut_void(),
            );

            hr.into_device_result("Zero buffer creation")?;

            let range = d3d12::D3D12_RANGE { Begin: 0, End: 0 };
            let mut ptr = std::ptr::null_mut();
            (*zero_buffer)
                .Map(0, &range, &mut ptr)
                .into_device_result("Map zero buffer")?;
            slice::from_raw_parts_mut(ptr as *mut u8, super::ZERO_BUFFER_SIZE as usize).fill(0);
            (*zero_buffer).Unmap(0, &range);
        };

        let shared = super::DeviceShared {
            zero_buffer,
            cmd_signatures: super::CommandSignatures {
                draw: raw
                    .create_command_signature(
                        native::RootSignature::null(),
                        &[native::IndirectArgument::draw()],
                        mem::size_of::<wgt::DrawIndirectArgs>() as u32,
                        0,
                    )
                    .into_device_result("Command (draw) signature creation")?,
                draw_indexed: raw
                    .create_command_signature(
                        native::RootSignature::null(),
                        &[native::IndirectArgument::draw_indexed()],
                        mem::size_of::<wgt::DrawIndexedIndirectArgs>() as u32,
                        0,
                    )
                    .into_device_result("Command (draw_indexed) signature creation")?,
                dispatch: raw
                    .create_command_signature(
                        native::RootSignature::null(),
                        &[native::IndirectArgument::dispatch()],
                        mem::size_of::<wgt::DispatchIndirectArgs>() as u32,
                        0,
                    )
                    .into_device_result("Command (dispatch) signature creation")?,
            },
        };

        Ok(super::Device {
            raw,
            present_queue,
            idler: super::Idler {
                fence: idle_fence,
                event: native::Event::create(false, false),
            },
            private_caps,
            shared: Arc::new(shared),
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
            sampler_pool: Mutex::new(descriptor::CpuPool::new(
                raw,
                native::DescriptorHeapType::Sampler,
            )),
            library: Arc::clone(library),
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
        hr.into_device_result("Set event")?;
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
        let array_size = match desc.range.array_layer_count {
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
                    ArraySize: array_size,
                }
            }
            wgt::TextureViewDimension::D2Array => {
                raw_desc.ViewDimension = d3d12::D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                *raw_desc.u.Texture2DArray_mut() = d3d12::D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: desc.range.base_mip_level,
                    MipLevels,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize: array_size,
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
                    NumCubes: array_size / 6,
                    ResourceMinLODClamp: 0.0,
                }
            }
        }

        let handle = self.srv_uav_pool.lock().alloc_handle();
        self.raw
            .CreateShaderResourceView(texture.resource.as_mut_ptr(), &raw_desc, handle.raw);
        handle
    }

    unsafe fn view_texture_as_unoredered_access(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> descriptor::Handle {
        let mut raw_desc = d3d12::D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: conv::map_texture_format(desc.format),
            ViewDimension: 0,
            u: mem::zeroed(),
        };

        let array_size = match desc.range.array_layer_count {
            Some(count) => count.get(),
            None => texture.size.depth_or_array_layers - desc.range.base_array_layer,
        };

        match desc.dimension {
            wgt::TextureViewDimension::D1 => {
                raw_desc.ViewDimension = d3d12::D3D12_UAV_DIMENSION_TEXTURE1D;
                *raw_desc.u.Texture1D_mut() = d3d12::D3D12_TEX1D_UAV {
                    MipSlice: desc.range.base_mip_level,
                }
            }
            /*
            wgt::TextureViewDimension::D1Array => {
                raw_desc.ViewDimension = d3d12::D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
                *raw_desc.u.Texture1DArray_mut() = d3d12::D3D12_TEX1D_ARRAY_UAV {
                    MipSlice: desc.range.base_mip_level,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize,
                }
            }*/
            wgt::TextureViewDimension::D2 => {
                raw_desc.ViewDimension = d3d12::D3D12_UAV_DIMENSION_TEXTURE2D;
                *raw_desc.u.Texture2D_mut() = d3d12::D3D12_TEX2D_UAV {
                    MipSlice: desc.range.base_mip_level,
                    PlaneSlice: 0,
                }
            }
            wgt::TextureViewDimension::D2Array => {
                raw_desc.ViewDimension = d3d12::D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                *raw_desc.u.Texture2DArray_mut() = d3d12::D3D12_TEX2D_ARRAY_UAV {
                    MipSlice: desc.range.base_mip_level,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize: array_size,
                    PlaneSlice: 0,
                }
            }
            wgt::TextureViewDimension::D3 => {
                raw_desc.ViewDimension = d3d12::D3D12_UAV_DIMENSION_TEXTURE3D;
                *raw_desc.u.Texture3D_mut() = d3d12::D3D12_TEX3D_UAV {
                    MipSlice: desc.range.base_mip_level,
                    FirstWSlice: desc.range.base_array_layer,
                    WSize: array_size,
                }
            }
            wgt::TextureViewDimension::Cube | wgt::TextureViewDimension::CubeArray => {
                panic!("Unable to view texture as cube UAV")
            }
        }

        let handle = self.srv_uav_pool.lock().alloc_handle();
        self.raw.CreateUnorderedAccessView(
            texture.resource.as_mut_ptr(),
            ptr::null_mut(),
            &raw_desc,
            handle.raw,
        );
        handle
    }

    unsafe fn view_texture_as_render_target(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> descriptor::Handle {
        let mut raw_desc = d3d12::D3D12_RENDER_TARGET_VIEW_DESC {
            Format: conv::map_texture_format(desc.format),
            ViewDimension: 0,
            u: mem::zeroed(),
        };

        let array_size = match desc.range.array_layer_count {
            Some(count) => count.get(),
            None => texture.size.depth_or_array_layers - desc.range.base_array_layer,
        };

        match desc.dimension {
            wgt::TextureViewDimension::D1 => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE1D;
                *raw_desc.u.Texture1D_mut() = d3d12::D3D12_TEX1D_RTV {
                    MipSlice: desc.range.base_mip_level,
                }
            }
            /*
            wgt::TextureViewDimension::D1Array => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
                *raw_desc.u.Texture1DArray_mut() = d3d12::D3D12_TEX1D_ARRAY_RTV {
                    MipSlice: desc.range.base_mip_level,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize,
                }
            }*/
            wgt::TextureViewDimension::D2 if texture.sample_count > 1 => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE2DMS;
                *raw_desc.u.Texture2DMS_mut() = d3d12::D3D12_TEX2DMS_RTV {
                    UnusedField_NothingToDefine: 0,
                }
            }
            wgt::TextureViewDimension::D2 => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE2D;
                *raw_desc.u.Texture2D_mut() = d3d12::D3D12_TEX2D_RTV {
                    MipSlice: desc.range.base_mip_level,
                    PlaneSlice: 0,
                }
            }
            wgt::TextureViewDimension::D2Array if texture.sample_count > 1 => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
                *raw_desc.u.Texture2DMSArray_mut() = d3d12::D3D12_TEX2DMS_ARRAY_RTV {
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize: array_size,
                }
            }
            wgt::TextureViewDimension::D2Array => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
                *raw_desc.u.Texture2DArray_mut() = d3d12::D3D12_TEX2D_ARRAY_RTV {
                    MipSlice: desc.range.base_mip_level,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize: array_size,
                    PlaneSlice: 0,
                }
            }
            wgt::TextureViewDimension::D3 => {
                raw_desc.ViewDimension = d3d12::D3D12_RTV_DIMENSION_TEXTURE3D;
                *raw_desc.u.Texture3D_mut() = d3d12::D3D12_TEX3D_RTV {
                    MipSlice: desc.range.base_mip_level,
                    FirstWSlice: desc.range.base_array_layer,
                    WSize: array_size,
                }
            }
            wgt::TextureViewDimension::Cube | wgt::TextureViewDimension::CubeArray => {
                panic!("Unable to view texture as cube RTV")
            }
        }

        let handle = self.rtv_pool.lock().alloc_handle();
        self.raw
            .CreateRenderTargetView(texture.resource.as_mut_ptr(), &raw_desc, handle.raw);
        handle
    }

    unsafe fn view_texture_as_depth_stencil(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
        read_only: bool,
    ) -> descriptor::Handle {
        let mut raw_desc = d3d12::D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: conv::map_texture_format(desc.format),
            ViewDimension: 0,
            Flags: if read_only {
                let aspects = crate::FormatAspects::from(desc.format);
                let mut flags = 0;
                if aspects.contains(crate::FormatAspects::DEPTH) {
                    flags |= d3d12::D3D12_DSV_FLAG_READ_ONLY_DEPTH;
                }
                if aspects.contains(crate::FormatAspects::STENCIL) {
                    flags |= d3d12::D3D12_DSV_FLAG_READ_ONLY_STENCIL;
                }
                flags
            } else {
                d3d12::D3D12_DSV_FLAG_NONE
            },
            u: mem::zeroed(),
        };

        let array_size = match desc.range.array_layer_count {
            Some(count) => count.get(),
            None => texture.size.depth_or_array_layers - desc.range.base_array_layer,
        };

        match desc.dimension {
            wgt::TextureViewDimension::D1 => {
                raw_desc.ViewDimension = d3d12::D3D12_DSV_DIMENSION_TEXTURE1D;
                *raw_desc.u.Texture1D_mut() = d3d12::D3D12_TEX1D_DSV {
                    MipSlice: desc.range.base_mip_level,
                }
            }
            /*
            wgt::TextureViewDimension::D1Array => {
                raw_desc.ViewDimension = d3d12::D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
                *raw_desc.u.Texture1DArray_mut() = d3d12::D3D12_TEX1D_ARRAY_DSV {
                    MipSlice: desc.range.base_mip_level,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize,
                }
            }*/
            wgt::TextureViewDimension::D2 if texture.sample_count > 1 => {
                raw_desc.ViewDimension = d3d12::D3D12_DSV_DIMENSION_TEXTURE2DMS;
                *raw_desc.u.Texture2DMS_mut() = d3d12::D3D12_TEX2DMS_DSV {
                    UnusedField_NothingToDefine: 0,
                }
            }
            wgt::TextureViewDimension::D2 => {
                raw_desc.ViewDimension = d3d12::D3D12_DSV_DIMENSION_TEXTURE2D;
                *raw_desc.u.Texture2D_mut() = d3d12::D3D12_TEX2D_DSV {
                    MipSlice: desc.range.base_mip_level,
                }
            }
            wgt::TextureViewDimension::D2Array if texture.sample_count > 1 => {
                raw_desc.ViewDimension = d3d12::D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
                *raw_desc.u.Texture2DMSArray_mut() = d3d12::D3D12_TEX2DMS_ARRAY_DSV {
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize: array_size,
                }
            }
            wgt::TextureViewDimension::D2Array => {
                raw_desc.ViewDimension = d3d12::D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
                *raw_desc.u.Texture2DArray_mut() = d3d12::D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: desc.range.base_mip_level,
                    FirstArraySlice: desc.range.base_array_layer,
                    ArraySize: array_size,
                }
            }
            wgt::TextureViewDimension::D3
            | wgt::TextureViewDimension::Cube
            | wgt::TextureViewDimension::CubeArray => {
                panic!("Unable to view texture as cube or 3D RTV")
            }
        }

        let handle = self.dsv_pool.lock().alloc_handle();
        self.raw
            .CreateDepthStencilView(texture.resource.as_mut_ptr(), &raw_desc, handle.raw);
        handle
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self) {
        self.rtv_pool.into_inner().destroy();
        self.dsv_pool.into_inner().destroy();
        self.srv_uav_pool.into_inner().destroy();
        self.sampler_pool.into_inner().destroy();
        self.shared.destroy();

        //self.descriptor_updater.lock().destroy();

        // Debug tracking alive objects
        if let Ok(debug_device) = self
            .raw
            .cast::<d3d12sdklayers::ID3D12DebugDevice>()
            .into_result()
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

        hr.into_device_result("Buffer creation")?;
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
        hr.into_device_result("Map buffer")?;
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
            let cwstr = conv::map_label(label);
            resource.SetName(cwstr.as_ptr());
        }

        hr.into_device_result("Texture creation")?;
        Ok(super::Texture {
            resource,
            format: desc.format,
            dimension: desc.dimension,
            size: desc.size,
            mip_level_count: desc.mip_level_count,
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
            handle_uav: if desc.usage.intersects(crate::TextureUses::STORAGE_STORE) {
                Some(self.view_texture_as_unoredered_access(texture, desc))
            } else {
                None
            },
            handle_rtv: if desc.usage.intersects(crate::TextureUses::COLOR_TARGET) {
                Some(self.view_texture_as_render_target(texture, desc))
            } else {
                None
            },
            handle_dsv_ro: if desc
                .usage
                .intersects(crate::TextureUses::DEPTH_STENCIL_READ)
            {
                Some(self.view_texture_as_depth_stencil(texture, desc, true))
            } else {
                None
            },
            handle_dsv_rw: if desc
                .usage
                .intersects(crate::TextureUses::DEPTH_STENCIL_WRITE)
            {
                Some(self.view_texture_as_depth_stencil(texture, desc, false))
            } else {
                None
            },
        })
    }
    unsafe fn destroy_texture_view(&self, view: super::TextureView) {
        if view.handle_srv.is_some() || view.handle_uav.is_some() {
            let mut pool = self.srv_uav_pool.lock();
            if let Some(handle) = view.handle_srv {
                pool.free_handle(handle);
            }
            if let Some(handle) = view.handle_uav {
                pool.free_handle(handle);
            }
        }
        if let Some(handle) = view.handle_rtv {
            self.rtv_pool.lock().free_handle(handle);
        }
        if view.handle_dsv_ro.is_some() || view.handle_dsv_rw.is_some() {
            let mut pool = self.dsv_pool.lock();
            if let Some(handle) = view.handle_dsv_ro {
                pool.free_handle(handle);
            }
            if let Some(handle) = view.handle_dsv_rw {
                pool.free_handle(handle);
            }
        }
    }

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        let handle = self.sampler_pool.lock().alloc_handle();

        let reduction = match desc.compare {
            Some(_) => d3d12::D3D12_FILTER_REDUCTION_TYPE_COMPARISON,
            None => d3d12::D3D12_FILTER_REDUCTION_TYPE_STANDARD,
        };
        let filter = conv::map_filter_mode(desc.min_filter) << d3d12::D3D12_MIN_FILTER_SHIFT
            | conv::map_filter_mode(desc.mag_filter) << d3d12::D3D12_MAG_FILTER_SHIFT
            | conv::map_filter_mode(desc.mipmap_filter) << d3d12::D3D12_MIP_FILTER_SHIFT
            | reduction << d3d12::D3D12_FILTER_REDUCTION_TYPE_SHIFT
            | desc
                .anisotropy_clamp
                .map_or(0, |_| d3d12::D3D12_FILTER_ANISOTROPIC);

        self.raw.create_sampler(
            handle.raw,
            filter,
            [
                conv::map_address_mode(desc.address_modes[0]),
                conv::map_address_mode(desc.address_modes[1]),
                conv::map_address_mode(desc.address_modes[2]),
            ],
            0.0,
            desc.anisotropy_clamp.map_or(0, |aniso| aniso.get() as u32),
            conv::map_comparison(desc.compare.unwrap_or(wgt::CompareFunction::Always)),
            conv::map_border_color(desc.border_color),
            desc.lod_clamp.clone().unwrap_or(0.0..16.0),
        );

        Ok(super::Sampler { handle })
    }
    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        self.sampler_pool.lock().free_handle(sampler.handle);
    }

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        let allocator = self
            .raw
            .create_command_allocator(native::CmdListType::Direct)
            .into_device_result("Command allocator creation")?;
        Ok(super::CommandEncoder {
            allocator,
            device: self.raw,
            shared: Arc::clone(&self.shared),
            list: None,
            free_lists: Vec::new(),
            temp: super::Temp::default(),
        })
    }
    unsafe fn destroy_command_encoder(&self, encoder: super::CommandEncoder) {
        if let Some(list) = encoder.list {
            list.close();
            list.destroy();
        }
        for list in encoder.free_lists {
            list.destroy();
        }
        encoder.allocator.destroy();
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<super::BindGroupLayout, crate::DeviceError> {
        Ok(super::BindGroupLayout {
            entries: desc.entries.to_vec(),
        })
    }
    unsafe fn destroy_bind_group_layout(&self, _bg_layout: super::BindGroupLayout) {
        // just drop
    }
    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> Result<super::PipelineLayout, crate::DeviceError> {
        // Pipeline layouts are implemented as RootSignature for D3D12.
        //
        // Push Constants are implemented as root constants.
        //
        // Each descriptor set layout will be one table entry of the root signature.
        // We have the additional restriction that SRV/CBV/UAV and samplers need to be
        // separated, so each set layout will actually occupy up to 2 entries!
        // SRV/CBV/UAV tables are added to the signature first, then Sampler tables,
        // and finally dynamic uniform descriptors.
        //
        // Dynamic uniform buffers are implemented as root descriptors.
        // This allows to handle the dynamic offsets properly, which would not be feasible
        // with a combination of root constant and descriptor table.
        //
        // Root signature layout:
        //     Root Constants: Register: Offest/4, Space: 0
        //     ...
        // DescriptorTable0: Space: 1 (SrvCbvUav)
        // DescriptorTable0: Space: 1 (Sampler)
        // Root Descriptors 0
        // DescriptorTable1: Space: 2 (SrvCbvUav)
        // Root Descriptors 1
        //     ...

        let mut root_offset = 0u32;
        let root_constants: &[()] = &[];

        // Number of elements in the root signature.
        let total_parameters = root_constants.len() + desc.bind_group_layouts.len() * 2;
        // Guarantees that no re-allocation is done, and our pointers are valid
        let mut parameters = Vec::with_capacity(total_parameters);
        let mut parameter_offsets = Vec::with_capacity(total_parameters);

        let root_space_offset = if !root_constants.is_empty() { 1 } else { 0 };
        // Collect the whole number of bindings we will create upfront.
        // It allows us to preallocate enough storage to avoid reallocation,
        // which could cause invalid pointers.
        let total_non_dynamic_entries = desc
            .bind_group_layouts
            .iter()
            .flat_map(|bgl| {
                bgl.entries.iter().map(|entry| match entry.ty {
                    wgt::BindingType::Buffer {
                        has_dynamic_offset: true,
                        ..
                    } => 0,
                    _ => 1,
                })
            })
            .sum();
        let mut ranges = Vec::with_capacity(total_non_dynamic_entries);

        let mut root_elements =
            arrayvec::ArrayVec::<[super::RootElement; crate::MAX_BIND_GROUPS]>::default();
        for (index, bgl) in desc.bind_group_layouts.iter().enumerate() {
            let space = root_space_offset + index as u32;
            let mut types = super::TableTypes::empty();
            let root_table_offset = root_offset as usize;

            let mut visibility_view_static = wgt::ShaderStages::empty();
            let mut visibility_view_dynamic = wgt::ShaderStages::empty();
            let mut visibility_sampler = wgt::ShaderStages::empty();
            for entry in bgl.entries.iter() {
                match entry.ty {
                    wgt::BindingType::Sampler { .. } => visibility_sampler |= entry.visibility,
                    wgt::BindingType::Buffer {
                        has_dynamic_offset: true,
                        ..
                    } => visibility_view_dynamic |= entry.visibility,
                    _ => visibility_view_static |= entry.visibility,
                }
            }

            // SRV/CBV/UAV descriptor tables
            let mut range_base = ranges.len();
            for entry in bgl.entries.iter() {
                let range_ty = match entry.ty {
                    wgt::BindingType::Buffer {
                        has_dynamic_offset: true,
                        ..
                    }
                    | wgt::BindingType::Sampler { .. } => continue,
                    ref other => conv::map_binding_type(other),
                };
                ranges.push(native::DescriptorRange::new(
                    range_ty,
                    entry.count.map_or(1, |count| count.get()),
                    native::Binding {
                        register: entry.binding,
                        space,
                    },
                    d3d12::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ));
            }
            if ranges.len() > range_base {
                parameter_offsets.push(root_offset);
                parameters.push(native::RootParameter::descriptor_table(
                    conv::map_visibility(visibility_view_static),
                    &ranges[range_base..],
                ));
                types |= super::TableTypes::SRV_CBV_UAV;
                root_offset += 1;
            }

            // Sampler descriptor tables
            range_base = ranges.len();
            for entry in bgl.entries.iter() {
                let range_ty = match entry.ty {
                    wgt::BindingType::Sampler { .. } => native::DescriptorRangeType::Sampler,
                    _ => continue,
                };
                ranges.push(native::DescriptorRange::new(
                    range_ty,
                    entry.count.map_or(1, |count| count.get()),
                    native::Binding {
                        register: entry.binding,
                        space,
                    },
                    d3d12::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ));
            }
            if ranges.len() > range_base {
                parameter_offsets.push(root_offset);
                parameters.push(native::RootParameter::descriptor_table(
                    conv::map_visibility(visibility_sampler),
                    &ranges[range_base..],
                ));
                types |= super::TableTypes::SAMPLERS;
                root_offset += 1;
            }

            // Root (dynamic) descriptor tables
            let dynamic_buffers_visibility = conv::map_visibility(visibility_view_dynamic);
            for entry in bgl.entries.iter() {
                let buffer_ty = match entry.ty {
                    wgt::BindingType::Buffer {
                        has_dynamic_offset: true,
                        ty,
                        ..
                    } => ty,
                    _ => continue,
                };
                let binding = native::Binding {
                    register: entry.binding,
                    space,
                };
                let param = match buffer_ty {
                    wgt::BufferBindingType::Uniform => {
                        native::RootParameter::cbv_descriptor(dynamic_buffers_visibility, binding)
                    }
                    wgt::BufferBindingType::Storage { read_only: true } => {
                        native::RootParameter::srv_descriptor(dynamic_buffers_visibility, binding)
                    }
                    wgt::BufferBindingType::Storage { read_only: false } => {
                        native::RootParameter::uav_descriptor(dynamic_buffers_visibility, binding)
                    }
                };
                parameter_offsets.push(root_offset);
                parameters.push(param);
                root_offset += 2; // root view costs 2 words
            }

            root_elements.push(super::RootElement {
                types,
                offset: root_table_offset,
            });
        }

        // Ensure that we didn't reallocate!
        debug_assert_eq!(ranges.len(), total_non_dynamic_entries);
        assert_eq!(parameters.len(), parameter_offsets.len());

        let (blob, error) = self
            .library
            .serialize_root_signature(
                native::RootSignatureVersion::V1_0,
                &parameters,
                &[],
                native::RootSignatureFlags::ALLOW_IA_INPUT_LAYOUT,
            )
            .map_err(|e| {
                log::error!("Unable to find serialization function: {:?}", e);
                crate::DeviceError::Lost
            })?
            .into_device_result("Root signature serialization")?;

        if !error.is_null() {
            log::error!(
                "Root signature serialization error: {:?}",
                error.as_c_str().to_str().unwrap()
            );
            error.destroy();
            return Err(crate::DeviceError::Lost);
        }

        let raw = self
            .raw
            .create_root_signature(blob, 0)
            .into_device_result("Root signature creation")?;
        blob.destroy();

        Ok(super::PipelineLayout {
            raw,
            parameter_offsets,
            total_slots: root_offset,
            elements: root_elements,
        })
    }
    unsafe fn destroy_pipeline_layout(&self, pipeline_layout: super::PipelineLayout) {
        pipeline_layout.raw.destroy();
    }

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> Result<super::BindGroup, crate::DeviceError> {
        Ok(super::BindGroup {})
    }
    unsafe fn destroy_bind_group(&self, group: super::BindGroup) {}

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
    ) -> Result<super::QuerySet, crate::DeviceError> {
        let (heap_ty, raw_ty) = match desc.ty {
            wgt::QueryType::Occlusion => (
                native::QueryHeapType::Occlusion,
                d3d12::D3D12_QUERY_TYPE_BINARY_OCCLUSION,
            ),
            wgt::QueryType::PipelineStatistics(_) => (
                native::QueryHeapType::PipelineStatistics,
                d3d12::D3D12_QUERY_TYPE_TIMESTAMP,
            ),
            wgt::QueryType::Timestamp => (
                native::QueryHeapType::Timestamp,
                d3d12::D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            ),
        };

        let raw = self
            .raw
            .create_query_heap(heap_ty, desc.count, 0)
            .into_device_result("Query heap creation")?;

        Ok(super::QuerySet { raw, raw_ty })
    }
    unsafe fn destroy_query_set(&self, set: super::QuerySet) {
        set.raw.destroy();
    }

    unsafe fn create_fence(&self) -> Result<super::Fence, crate::DeviceError> {
        let mut raw = native::Fence::null();
        let hr = self.raw.CreateFence(
            0,
            d3d12::D3D12_FENCE_FLAG_NONE,
            &d3d12::ID3D12Fence::uuidof(),
            raw.mut_void(),
        );
        hr.into_device_result("Fence creation")?;
        Ok(super::Fence { raw })
    }
    unsafe fn destroy_fence(&self, fence: super::Fence) {
        fence.raw.destroy();
    }
    unsafe fn get_fence_value(
        &self,
        fence: &super::Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        Ok(fence.raw.GetCompletedValue())
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        if fence.raw.GetCompletedValue() >= value {
            return Ok(true);
        }
        let hr = fence.raw.set_event_on_completion(self.idler.event, value);
        hr.into_device_result("Set event")?;

        match synchapi::WaitForSingleObject(self.idler.event.0, timeout_ms) {
            winbase::WAIT_ABANDONED | winbase::WAIT_FAILED => Err(crate::DeviceError::Lost),
            winbase::WAIT_OBJECT_0 => Ok(true),
            winerror::WAIT_TIMEOUT => Ok(false),
            other => {
                log::error!("Unexpected wait status: 0x{:x}", other);
                Err(crate::DeviceError::Lost)
            }
        }
    }

    unsafe fn start_capture(&self) -> bool {
        false
    }
    unsafe fn stop_capture(&self) {}
}
