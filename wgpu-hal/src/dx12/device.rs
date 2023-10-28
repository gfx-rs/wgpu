use crate::{
    auxil::{self, dxgi::result::HResult as _},
    dx12::shader_compilation,
};

use super::{conv, descriptor, view};
use parking_lot::Mutex;
use std::{ffi, mem, num::NonZeroU32, ptr, sync::Arc};
use winapi::{
    shared::{dxgiformat, dxgitype, minwindef::BOOL, winerror},
    um::{d3d12 as d3d12_ty, synchapi, winbase},
    Interface,
};

// this has to match Naga's HLSL backend, and also needs to be null-terminated
const NAGA_LOCATION_SEMANTIC: &[u8] = b"LOC\0";

impl super::Device {
    pub(super) fn new(
        raw: d3d12::Device,
        present_queue: d3d12::CommandQueue,
        limits: &wgt::Limits,
        private_caps: super::PrivateCapabilities,
        library: &Arc<d3d12::D3D12Lib>,
        dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
    ) -> Result<Self, crate::DeviceError> {
        let mem_allocator = if private_caps.suballocation_supported {
            super::suballocation::create_allocator_wrapper(&raw)?
        } else {
            None
        };

        let mut idle_fence = d3d12::Fence::null();
        let hr = unsafe {
            profiling::scope!("ID3D12Device::CreateFence");
            raw.CreateFence(
                0,
                d3d12_ty::D3D12_FENCE_FLAG_NONE,
                &d3d12_ty::ID3D12Fence::uuidof(),
                idle_fence.mut_void(),
            )
        };
        hr.into_device_result("Idle fence creation")?;

        let mut zero_buffer = d3d12::Resource::null();
        unsafe {
            let raw_desc = d3d12_ty::D3D12_RESOURCE_DESC {
                Dimension: d3d12_ty::D3D12_RESOURCE_DIMENSION_BUFFER,
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
                Layout: d3d12_ty::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: d3d12_ty::D3D12_RESOURCE_FLAG_NONE,
            };

            let heap_properties = d3d12_ty::D3D12_HEAP_PROPERTIES {
                Type: d3d12_ty::D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: d3d12_ty::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
                MemoryPoolPreference: match private_caps.memory_architecture {
                    super::MemoryArchitecture::Unified { .. } => d3d12_ty::D3D12_MEMORY_POOL_L0,
                    super::MemoryArchitecture::NonUnified => d3d12_ty::D3D12_MEMORY_POOL_L1,
                },
                CreationNodeMask: 0,
                VisibleNodeMask: 0,
            };

            profiling::scope!("Zero Buffer Allocation");
            raw.CreateCommittedResource(
                &heap_properties,
                d3d12_ty::D3D12_HEAP_FLAG_NONE,
                &raw_desc,
                d3d12_ty::D3D12_RESOURCE_STATE_COMMON,
                ptr::null(),
                &d3d12_ty::ID3D12Resource::uuidof(),
                zero_buffer.mut_void(),
            )
            .into_device_result("Zero buffer creation")?;

            // Note: without `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED`
            // this resource is zeroed by default.
        };

        // maximum number of CBV/SRV/UAV descriptors in heap for Tier 1
        let capacity_views = limits.max_non_sampler_bindings as u64;
        let capacity_samplers = 2_048;

        let shared = super::DeviceShared {
            zero_buffer,
            cmd_signatures: super::CommandSignatures {
                draw: raw
                    .create_command_signature(
                        d3d12::RootSignature::null(),
                        &[d3d12::IndirectArgument::draw()],
                        mem::size_of::<wgt::DrawIndirectArgs>() as u32,
                        0,
                    )
                    .into_device_result("Command (draw) signature creation")?,
                draw_indexed: raw
                    .create_command_signature(
                        d3d12::RootSignature::null(),
                        &[d3d12::IndirectArgument::draw_indexed()],
                        mem::size_of::<wgt::DrawIndexedIndirectArgs>() as u32,
                        0,
                    )
                    .into_device_result("Command (draw_indexed) signature creation")?,
                dispatch: raw
                    .create_command_signature(
                        d3d12::RootSignature::null(),
                        &[d3d12::IndirectArgument::dispatch()],
                        mem::size_of::<wgt::DispatchIndirectArgs>() as u32,
                        0,
                    )
                    .into_device_result("Command (dispatch) signature creation")?,
            },
            heap_views: descriptor::GeneralHeap::new(
                raw.clone(),
                d3d12::DescriptorHeapType::CbvSrvUav,
                capacity_views,
            )?,
            heap_samplers: descriptor::GeneralHeap::new(
                raw.clone(),
                d3d12::DescriptorHeapType::Sampler,
                capacity_samplers,
            )?,
        };

        let mut rtv_pool = descriptor::CpuPool::new(raw.clone(), d3d12::DescriptorHeapType::Rtv);
        let null_rtv_handle = rtv_pool.alloc_handle();
        // A null pResource is used to initialize a null descriptor,
        // which guarantees D3D11-like null binding behavior (reading 0s, writes are discarded)
        raw.create_render_target_view(
            d3d12::ComPtr::null(),
            &d3d12::RenderTargetViewDesc::texture_2d(
                winapi::shared::dxgiformat::DXGI_FORMAT_R8G8B8A8_UNORM,
                0,
                0,
            ),
            null_rtv_handle.raw,
        );

        Ok(super::Device {
            raw: raw.clone(),
            present_queue,
            idler: super::Idler {
                fence: idle_fence,
                event: d3d12::Event::create(false, false),
            },
            private_caps,
            shared: Arc::new(shared),
            rtv_pool: Mutex::new(rtv_pool),
            dsv_pool: Mutex::new(descriptor::CpuPool::new(
                raw.clone(),
                d3d12::DescriptorHeapType::Dsv,
            )),
            srv_uav_pool: Mutex::new(descriptor::CpuPool::new(
                raw.clone(),
                d3d12::DescriptorHeapType::CbvSrvUav,
            )),
            sampler_pool: Mutex::new(descriptor::CpuPool::new(
                raw,
                d3d12::DescriptorHeapType::Sampler,
            )),
            library: Arc::clone(library),
            #[cfg(feature = "renderdoc")]
            render_doc: Default::default(),
            null_rtv_handle,
            mem_allocator,
            dxc_container,
        })
    }

    // Blocks until the dedicated present queue is finished with all of its work.
    //
    // Once this method completes, the surface is able to be resized or deleted.
    pub(super) unsafe fn wait_for_present_queue_idle(&self) -> Result<(), crate::DeviceError> {
        let cur_value = self.idler.fence.get_value();
        if cur_value == !0 {
            return Err(crate::DeviceError::Lost);
        }

        let value = cur_value + 1;
        log::info!("Waiting for idle with value {}", value);
        self.present_queue.signal(&self.idler.fence, value);
        let hr = self
            .idler
            .fence
            .set_event_on_completion(self.idler.event, value);
        hr.into_device_result("Set event")?;
        unsafe { synchapi::WaitForSingleObject(self.idler.event.0, winbase::INFINITE) };
        Ok(())
    }

    fn load_shader(
        &self,
        stage: &crate::ProgrammableStage<super::Api>,
        layout: &super::PipelineLayout,
        naga_stage: naga::ShaderStage,
    ) -> Result<super::CompiledShader, crate::PipelineError> {
        use naga::back::hlsl;

        let stage_bit = crate::auxil::map_naga_stage(naga_stage);
        let module = &stage.module.naga.module;
        //TODO: reuse the writer
        let mut source = String::new();
        let mut writer = hlsl::Writer::new(&mut source, &layout.naga_options);
        let reflection_info = {
            profiling::scope!("naga::back::hlsl::write");
            writer
                .write(module, &stage.module.naga.info)
                .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("HLSL: {e:?}")))?
        };

        let full_stage = format!(
            "{}_{}\0",
            naga_stage.to_hlsl_str(),
            layout.naga_options.shader_model.to_str()
        );

        let ep_index = module
            .entry_points
            .iter()
            .position(|ep| ep.stage == naga_stage && ep.name == stage.entry_point)
            .ok_or(crate::PipelineError::EntryPoint(naga_stage))?;

        let raw_ep = reflection_info.entry_point_names[ep_index]
            .as_ref()
            .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("{e}")))?;

        let source_name = stage
            .module
            .raw_name
            .as_ref()
            .and_then(|cstr| cstr.to_str().ok())
            .unwrap_or_default();

        // Compile with DXC if available, otherwise fall back to FXC
        let (result, log_level) = if let Some(ref dxc_container) = self.dxc_container {
            super::shader_compilation::compile_dxc(
                self,
                &source,
                source_name,
                raw_ep,
                stage_bit,
                full_stage,
                dxc_container,
            )
        } else {
            super::shader_compilation::compile_fxc(
                self,
                &source,
                source_name,
                &ffi::CString::new(raw_ep.as_str()).unwrap(),
                stage_bit,
                full_stage,
            )
        };

        log::log!(
            log_level,
            "Naga generated shader for {:?} at {:?}:\n{}",
            raw_ep,
            naga_stage,
            source
        );
        result
    }

    pub fn raw_device(&self) -> &d3d12::Device {
        &self.raw
    }

    pub fn raw_queue(&self) -> &d3d12::CommandQueue {
        &self.present_queue
    }

    pub unsafe fn texture_from_raw(
        resource: d3d12::Resource,
        format: wgt::TextureFormat,
        dimension: wgt::TextureDimension,
        size: wgt::Extent3d,
        mip_level_count: u32,
        sample_count: u32,
    ) -> super::Texture {
        super::Texture {
            resource,
            format,
            dimension,
            size,
            mip_level_count,
            sample_count,
            allocation: None,
        }
    }

    pub unsafe fn buffer_from_raw(
        resource: d3d12::Resource,
        size: wgt::BufferAddress,
    ) -> super::Buffer {
        super::Buffer {
            resource,
            size,
            allocation: None,
        }
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(mut self, _queue: super::Queue) {
        self.rtv_pool.lock().free_handle(self.null_rtv_handle);
        self.mem_allocator = None;
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let mut resource = d3d12::Resource::null();
        let mut size = desc.size;
        if desc.usage.contains(crate::BufferUses::UNIFORM) {
            let align_mask = d3d12_ty::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as u64 - 1;
            size = ((size - 1) | align_mask) + 1;
        }

        let raw_desc = d3d12_ty::D3D12_RESOURCE_DESC {
            Dimension: d3d12_ty::D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: dxgiformat::DXGI_FORMAT_UNKNOWN,
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: d3d12_ty::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: conv::map_buffer_usage_to_resource_flags(desc.usage),
        };

        let (hr, allocation) =
            super::suballocation::create_buffer_resource(self, desc, raw_desc, &mut resource)?;

        hr.into_device_result("Buffer creation")?;
        if let Some(label) = desc.label {
            let cwstr = conv::map_label(label);
            unsafe { resource.SetName(cwstr.as_ptr()) };
        }

        Ok(super::Buffer {
            resource,
            size,
            allocation,
        })
    }

    unsafe fn destroy_buffer(&self, mut buffer: super::Buffer) {
        // Only happens when it's using the windows_rs feature and there's an allocation
        if let Some(alloc) = buffer.allocation.take() {
            super::suballocation::free_buffer_allocation(
                alloc,
                // SAFETY: for allocations to exist, the allocator must exist
                unsafe { self.mem_allocator.as_ref().unwrap_unchecked() },
            );
        }
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let mut ptr = ptr::null_mut();
        // TODO: 0 for subresource should be fine here until map and unmap buffer is subresource aware?
        let hr = unsafe { (*buffer.resource).Map(0, ptr::null(), &mut ptr) };
        hr.into_device_result("Map buffer")?;
        Ok(crate::BufferMapping {
            ptr: ptr::NonNull::new(unsafe { ptr.offset(range.start as isize).cast::<u8>() })
                .unwrap(),
            //TODO: double-check this. Documentation is a bit misleading -
            // it implies that Map/Unmap is needed to invalidate/flush memory.
            is_coherent: true,
        })
    }

    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        unsafe { (*buffer.resource).Unmap(0, ptr::null()) };
        Ok(())
    }

    unsafe fn flush_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        use super::suballocation::create_texture_resource;

        let mut resource = d3d12::Resource::null();

        let raw_desc = d3d12_ty::D3D12_RESOURCE_DESC {
            Dimension: conv::map_texture_dimension(desc.dimension),
            Alignment: 0,
            Width: desc.size.width as u64,
            Height: desc.size.height,
            DepthOrArraySize: desc.size.depth_or_array_layers as u16,
            MipLevels: desc.mip_level_count as u16,
            Format: auxil::dxgi::conv::map_texture_format_for_resource(
                desc.format,
                desc.usage,
                !desc.view_formats.is_empty(),
                self.private_caps.casting_fully_typed_format_supported,
            ),
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: desc.sample_count,
                Quality: 0,
            },
            Layout: d3d12_ty::D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: conv::map_texture_usage_to_resource_flags(desc.usage),
        };

        let (hr, allocation) = create_texture_resource(self, desc, raw_desc, &mut resource)?;

        hr.into_device_result("Texture creation")?;
        if let Some(label) = desc.label {
            let cwstr = conv::map_label(label);
            unsafe { resource.SetName(cwstr.as_ptr()) };
        }

        Ok(super::Texture {
            resource,
            format: desc.format,
            dimension: desc.dimension,
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            allocation,
        })
    }

    unsafe fn destroy_texture(&self, mut texture: super::Texture) {
        if let Some(alloc) = texture.allocation.take() {
            super::suballocation::free_texture_allocation(
                alloc,
                // SAFETY: for allocations to exist, the allocator must exist
                unsafe { self.mem_allocator.as_ref().unwrap_unchecked() },
            );
        }
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        let view_desc = desc.to_internal(texture);

        Ok(super::TextureView {
            raw_format: view_desc.rtv_dsv_format,
            aspects: view_desc.aspects,
            target_base: (
                texture.resource.clone(),
                texture.calc_subresource(desc.range.base_mip_level, desc.range.base_array_layer, 0),
            ),
            handle_srv: if desc.usage.intersects(crate::TextureUses::RESOURCE) {
                let raw_desc = unsafe { view_desc.to_srv() };
                raw_desc.map(|raw_desc| {
                    let handle = self.srv_uav_pool.lock().alloc_handle();
                    unsafe {
                        self.raw.CreateShaderResourceView(
                            texture.resource.as_mut_ptr(),
                            &raw_desc,
                            handle.raw,
                        )
                    };
                    handle
                })
            } else {
                None
            },
            handle_uav: if desc.usage.intersects(
                crate::TextureUses::STORAGE_READ | crate::TextureUses::STORAGE_READ_WRITE,
            ) {
                let raw_desc = unsafe { view_desc.to_uav() };
                raw_desc.map(|raw_desc| {
                    let handle = self.srv_uav_pool.lock().alloc_handle();
                    unsafe {
                        self.raw.CreateUnorderedAccessView(
                            texture.resource.as_mut_ptr(),
                            ptr::null_mut(),
                            &raw_desc,
                            handle.raw,
                        )
                    };
                    handle
                })
            } else {
                None
            },
            handle_rtv: if desc.usage.intersects(crate::TextureUses::COLOR_TARGET) {
                let raw_desc = unsafe { view_desc.to_rtv() };
                let handle = self.rtv_pool.lock().alloc_handle();
                unsafe {
                    self.raw.CreateRenderTargetView(
                        texture.resource.as_mut_ptr(),
                        &raw_desc,
                        handle.raw,
                    )
                };
                Some(handle)
            } else {
                None
            },
            handle_dsv_ro: if desc
                .usage
                .intersects(crate::TextureUses::DEPTH_STENCIL_READ)
            {
                let raw_desc = unsafe { view_desc.to_dsv(true) };
                let handle = self.dsv_pool.lock().alloc_handle();
                unsafe {
                    self.raw.CreateDepthStencilView(
                        texture.resource.as_mut_ptr(),
                        &raw_desc,
                        handle.raw,
                    )
                };
                Some(handle)
            } else {
                None
            },
            handle_dsv_rw: if desc
                .usage
                .intersects(crate::TextureUses::DEPTH_STENCIL_WRITE)
            {
                let raw_desc = unsafe { view_desc.to_dsv(false) };
                let handle = self.dsv_pool.lock().alloc_handle();
                unsafe {
                    self.raw.CreateDepthStencilView(
                        texture.resource.as_mut_ptr(),
                        &raw_desc,
                        handle.raw,
                    )
                };
                Some(handle)
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
            Some(_) => d3d12_ty::D3D12_FILTER_REDUCTION_TYPE_COMPARISON,
            None => d3d12_ty::D3D12_FILTER_REDUCTION_TYPE_STANDARD,
        };
        let mut filter = conv::map_filter_mode(desc.min_filter) << d3d12_ty::D3D12_MIN_FILTER_SHIFT
            | conv::map_filter_mode(desc.mag_filter) << d3d12_ty::D3D12_MAG_FILTER_SHIFT
            | conv::map_filter_mode(desc.mipmap_filter) << d3d12_ty::D3D12_MIP_FILTER_SHIFT
            | reduction << d3d12_ty::D3D12_FILTER_REDUCTION_TYPE_SHIFT;

        if desc.anisotropy_clamp != 1 {
            filter |= d3d12_ty::D3D12_FILTER_ANISOTROPIC;
        };

        let border_color = conv::map_border_color(desc.border_color);

        self.raw.create_sampler(
            handle.raw,
            filter,
            [
                conv::map_address_mode(desc.address_modes[0]),
                conv::map_address_mode(desc.address_modes[1]),
                conv::map_address_mode(desc.address_modes[2]),
            ],
            0.0,
            desc.anisotropy_clamp as u32,
            conv::map_comparison(desc.compare.unwrap_or(wgt::CompareFunction::Always)),
            border_color,
            desc.lod_clamp.clone(),
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
            .create_command_allocator(d3d12::CmdListType::Direct)
            .into_device_result("Command allocator creation")?;

        if let Some(label) = desc.label {
            let cwstr = conv::map_label(label);
            unsafe { allocator.SetName(cwstr.as_ptr()) };
        }

        Ok(super::CommandEncoder {
            allocator,
            device: self.raw.clone(),
            shared: Arc::clone(&self.shared),
            null_rtv_handle: self.null_rtv_handle,
            list: None,
            free_lists: Vec::new(),
            pass: super::PassState::new(),
            temp: super::Temp::default(),
            end_of_pass_timer_query: None,
        })
    }
    unsafe fn destroy_command_encoder(&self, encoder: super::CommandEncoder) {
        if let Some(list) = encoder.list {
            list.close();
        }
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<super::BindGroupLayout, crate::DeviceError> {
        let (mut num_buffer_views, mut num_samplers, mut num_texture_views) = (0, 0, 0);
        for entry in desc.entries.iter() {
            let count = entry.count.map_or(1, NonZeroU32::get);
            match entry.ty {
                wgt::BindingType::Buffer {
                    has_dynamic_offset: true,
                    ..
                } => {}
                wgt::BindingType::Buffer { .. } => num_buffer_views += count,
                wgt::BindingType::Texture { .. } | wgt::BindingType::StorageTexture { .. } => {
                    num_texture_views += count
                }
                wgt::BindingType::Sampler { .. } => num_samplers += count,
            }
        }

        let num_views = num_buffer_views + num_texture_views;
        Ok(super::BindGroupLayout {
            entries: desc.entries.to_vec(),
            cpu_heap_views: if num_views != 0 {
                let heap = descriptor::CpuHeap::new(
                    self.raw.clone(),
                    d3d12::DescriptorHeapType::CbvSrvUav,
                    num_views,
                )?;
                Some(heap)
            } else {
                None
            },
            cpu_heap_samplers: if num_samplers != 0 {
                let heap = descriptor::CpuHeap::new(
                    self.raw.clone(),
                    d3d12::DescriptorHeapType::Sampler,
                    num_samplers,
                )?;
                Some(heap)
            } else {
                None
            },
            copy_counts: vec![1; num_views.max(num_samplers) as usize],
        })
    }
    unsafe fn destroy_bind_group_layout(&self, _bg_layout: super::BindGroupLayout) {}

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> Result<super::PipelineLayout, crate::DeviceError> {
        use naga::back::hlsl;
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
        // Buffers with dynamic offsets are implemented as root descriptors.
        // This is easier than trying to patch up the offset on the shader side.
        //
        // Root signature layout:
        // Root Constants: Parameter=0, Space=0
        //     ...
        // (bind group [0]) - Space=0
        //   View descriptor table, if any
        //   Sampler descriptor table, if any
        //   Root descriptors (for dynamic offset buffers)
        // (bind group [1]) - Space=0
        // ...
        // (bind group [2]) - Space=0
        // Special constant buffer: Space=0

        //TODO: put lower bind group indices futher down the root signature. See:
        // https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#binding-model
        // Currently impossible because wgpu-core only re-binds the descriptor sets based
        // on Vulkan-like layout compatibility rules.

        fn native_binding(bt: &hlsl::BindTarget) -> d3d12::Binding {
            d3d12::Binding {
                space: bt.space as u32,
                register: bt.register,
            }
        }

        log::debug!(
            "Creating Root Signature '{}'",
            desc.label.unwrap_or_default()
        );

        let mut binding_map = hlsl::BindingMap::default();
        let (mut bind_cbv, mut bind_srv, mut bind_uav, mut bind_sampler) = (
            hlsl::BindTarget::default(),
            hlsl::BindTarget::default(),
            hlsl::BindTarget::default(),
            hlsl::BindTarget::default(),
        );
        let mut parameters = Vec::new();
        let mut push_constants_target = None;
        let mut root_constant_info = None;

        let mut pc_start = u32::MAX;
        let mut pc_end = u32::MIN;

        for pc in desc.push_constant_ranges.iter() {
            pc_start = pc_start.min(pc.range.start);
            pc_end = pc_end.max(pc.range.end);
        }

        if pc_start != u32::MAX && pc_end != u32::MIN {
            let parameter_index = parameters.len();
            let size = (pc_end - pc_start) / 4;
            log::debug!(
                "\tParam[{}] = push constant (count = {})",
                parameter_index,
                size,
            );
            parameters.push(d3d12::RootParameter::constants(
                d3d12::ShaderVisibility::All,
                native_binding(&bind_cbv),
                size,
            ));
            let binding = bind_cbv.clone();
            bind_cbv.register += 1;
            root_constant_info = Some(super::RootConstantInfo {
                root_index: parameter_index as u32,
                range: (pc_start / 4)..(pc_end / 4),
            });
            push_constants_target = Some(binding);

            bind_cbv.space += 1;
        }

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

        let mut bind_group_infos =
            arrayvec::ArrayVec::<super::BindGroupInfo, { crate::MAX_BIND_GROUPS }>::default();
        for (index, bgl) in desc.bind_group_layouts.iter().enumerate() {
            let mut info = super::BindGroupInfo {
                tables: super::TableTypes::empty(),
                base_root_index: parameters.len() as u32,
                dynamic_buffers: Vec::new(),
            };

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
                    } => continue,
                    ref other => conv::map_binding_type(other),
                };
                let bt = match range_ty {
                    d3d12::DescriptorRangeType::CBV => &mut bind_cbv,
                    d3d12::DescriptorRangeType::SRV => &mut bind_srv,
                    d3d12::DescriptorRangeType::UAV => &mut bind_uav,
                    d3d12::DescriptorRangeType::Sampler => continue,
                };

                binding_map.insert(
                    naga::ResourceBinding {
                        group: index as u32,
                        binding: entry.binding,
                    },
                    hlsl::BindTarget {
                        binding_array_size: entry.count.map(NonZeroU32::get),
                        ..bt.clone()
                    },
                );
                ranges.push(d3d12::DescriptorRange::new(
                    range_ty,
                    entry.count.map_or(1, |count| count.get()),
                    native_binding(bt),
                    d3d12_ty::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ));
                bt.register += entry.count.map(NonZeroU32::get).unwrap_or(1);
            }
            if ranges.len() > range_base {
                log::debug!(
                    "\tParam[{}] = views (vis = {:?}, count = {})",
                    parameters.len(),
                    visibility_view_static,
                    ranges.len() - range_base,
                );
                parameters.push(d3d12::RootParameter::descriptor_table(
                    conv::map_visibility(visibility_view_static),
                    &ranges[range_base..],
                ));
                info.tables |= super::TableTypes::SRV_CBV_UAV;
            }

            // Sampler descriptor tables
            range_base = ranges.len();
            for entry in bgl.entries.iter() {
                let range_ty = match entry.ty {
                    wgt::BindingType::Sampler { .. } => d3d12::DescriptorRangeType::Sampler,
                    _ => continue,
                };
                binding_map.insert(
                    naga::ResourceBinding {
                        group: index as u32,
                        binding: entry.binding,
                    },
                    hlsl::BindTarget {
                        binding_array_size: entry.count.map(NonZeroU32::get),
                        ..bind_sampler.clone()
                    },
                );
                ranges.push(d3d12::DescriptorRange::new(
                    range_ty,
                    entry.count.map_or(1, |count| count.get()),
                    native_binding(&bind_sampler),
                    d3d12_ty::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ));
                bind_sampler.register += entry.count.map(NonZeroU32::get).unwrap_or(1);
            }
            if ranges.len() > range_base {
                log::debug!(
                    "\tParam[{}] = samplers (vis = {:?}, count = {})",
                    parameters.len(),
                    visibility_sampler,
                    ranges.len() - range_base,
                );
                parameters.push(d3d12::RootParameter::descriptor_table(
                    conv::map_visibility(visibility_sampler),
                    &ranges[range_base..],
                ));
                info.tables |= super::TableTypes::SAMPLERS;
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

                let (kind, parameter_ty, bt) = match buffer_ty {
                    wgt::BufferBindingType::Uniform => (
                        super::BufferViewKind::Constant,
                        d3d12_ty::D3D12_ROOT_PARAMETER_TYPE_CBV,
                        &mut bind_cbv,
                    ),
                    wgt::BufferBindingType::Storage { read_only: true } => (
                        super::BufferViewKind::ShaderResource,
                        d3d12_ty::D3D12_ROOT_PARAMETER_TYPE_SRV,
                        &mut bind_srv,
                    ),
                    wgt::BufferBindingType::Storage { read_only: false } => (
                        super::BufferViewKind::UnorderedAccess,
                        d3d12_ty::D3D12_ROOT_PARAMETER_TYPE_UAV,
                        &mut bind_uav,
                    ),
                };

                binding_map.insert(
                    naga::ResourceBinding {
                        group: index as u32,
                        binding: entry.binding,
                    },
                    hlsl::BindTarget {
                        binding_array_size: entry.count.map(NonZeroU32::get),
                        ..bt.clone()
                    },
                );
                info.dynamic_buffers.push(kind);

                log::debug!(
                    "\tParam[{}] = dynamic {:?} (vis = {:?})",
                    parameters.len(),
                    buffer_ty,
                    dynamic_buffers_visibility,
                );
                parameters.push(d3d12::RootParameter::descriptor(
                    parameter_ty,
                    dynamic_buffers_visibility,
                    native_binding(bt),
                ));

                bt.register += entry.count.map_or(1, NonZeroU32::get);
            }

            bind_group_infos.push(info);
        }

        // Ensure that we didn't reallocate!
        debug_assert_eq!(ranges.len(), total_non_dynamic_entries);

        let (special_constants_root_index, special_constants_binding) = if desc.flags.intersects(
            crate::PipelineLayoutFlags::BASE_VERTEX_INSTANCE
                | crate::PipelineLayoutFlags::NUM_WORK_GROUPS,
        ) {
            let parameter_index = parameters.len();
            log::debug!("\tParam[{}] = special", parameter_index);
            parameters.push(d3d12::RootParameter::constants(
                d3d12::ShaderVisibility::All, // really needed for VS and CS only
                native_binding(&bind_cbv),
                3, // 0 = base vertex, 1 = base instance, 2 = other
            ));
            let binding = bind_cbv.clone();
            bind_cbv.register += 1;
            (Some(parameter_index as u32), Some(binding))
        } else {
            (None, None)
        };

        log::trace!("{:#?}", parameters);
        log::trace!("Bindings {:#?}", binding_map);

        let (blob, error) = self
            .library
            .serialize_root_signature(
                d3d12::RootSignatureVersion::V1_0,
                &parameters,
                &[],
                d3d12::RootSignatureFlags::ALLOW_IA_INPUT_LAYOUT,
            )
            .map_err(|e| {
                log::error!("Unable to find serialization function: {:?}", e);
                crate::DeviceError::Lost
            })?
            .into_device_result("Root signature serialization")?;

        if !error.is_null() {
            log::error!(
                "Root signature serialization error: {:?}",
                unsafe { error.as_c_str() }.to_str().unwrap()
            );
            return Err(crate::DeviceError::Lost);
        }

        let raw = self
            .raw
            .create_root_signature(blob, 0)
            .into_device_result("Root signature creation")?;

        log::debug!("\traw = {:?}", raw);

        if let Some(label) = desc.label {
            let cwstr = conv::map_label(label);
            unsafe { raw.SetName(cwstr.as_ptr()) };
        }

        Ok(super::PipelineLayout {
            shared: super::PipelineLayoutShared {
                signature: raw,
                total_root_elements: parameters.len() as super::RootIndex,
                special_constants_root_index,
                root_constant_info,
            },
            bind_group_infos,
            naga_options: hlsl::Options {
                shader_model: match self.dxc_container {
                    // DXC
                    Some(_) => hlsl::ShaderModel::V6_0,
                    // FXC doesn't support SM 6.0
                    None => hlsl::ShaderModel::V5_1,
                },
                binding_map,
                fake_missing_bindings: false,
                special_constants_binding,
                push_constants_target,
                zero_initialize_workgroup_memory: true,
            },
        })
    }
    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: super::PipelineLayout) {}

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> Result<super::BindGroup, crate::DeviceError> {
        let mut cpu_views = desc
            .layout
            .cpu_heap_views
            .as_ref()
            .map(|cpu_heap| cpu_heap.inner.lock());
        if let Some(ref mut inner) = cpu_views {
            inner.stage.clear();
        }
        let mut cpu_samplers = desc
            .layout
            .cpu_heap_samplers
            .as_ref()
            .map(|cpu_heap| cpu_heap.inner.lock());
        if let Some(ref mut inner) = cpu_samplers {
            inner.stage.clear();
        }
        let mut dynamic_buffers = Vec::new();

        for (layout, entry) in desc.layout.entries.iter().zip(desc.entries.iter()) {
            match layout.ty {
                wgt::BindingType::Buffer {
                    has_dynamic_offset: true,
                    ..
                } => {
                    let start = entry.resource_index as usize;
                    let end = start + entry.count as usize;
                    for data in &desc.buffers[start..end] {
                        dynamic_buffers.push(data.resolve_address());
                    }
                }
                wgt::BindingType::Buffer { ty, .. } => {
                    let start = entry.resource_index as usize;
                    let end = start + entry.count as usize;
                    for data in &desc.buffers[start..end] {
                        let gpu_address = data.resolve_address();
                        let size = data.resolve_size() as u32;
                        let inner = cpu_views.as_mut().unwrap();
                        let cpu_index = inner.stage.len() as u32;
                        let handle = desc.layout.cpu_heap_views.as_ref().unwrap().at(cpu_index);
                        match ty {
                            wgt::BufferBindingType::Uniform => {
                                let size_mask =
                                    d3d12_ty::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1;
                                let raw_desc = d3d12_ty::D3D12_CONSTANT_BUFFER_VIEW_DESC {
                                    BufferLocation: gpu_address,
                                    SizeInBytes: ((size - 1) | size_mask) + 1,
                                };
                                unsafe { self.raw.CreateConstantBufferView(&raw_desc, handle) };
                            }
                            wgt::BufferBindingType::Storage { read_only: true } => {
                                let mut raw_desc = d3d12_ty::D3D12_SHADER_RESOURCE_VIEW_DESC {
                                    Format: dxgiformat::DXGI_FORMAT_R32_TYPELESS,
                                    Shader4ComponentMapping:
                                        view::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                                    ViewDimension: d3d12_ty::D3D12_SRV_DIMENSION_BUFFER,
                                    u: unsafe { mem::zeroed() },
                                };
                                unsafe {
                                    *raw_desc.u.Buffer_mut() = d3d12_ty::D3D12_BUFFER_SRV {
                                        FirstElement: data.offset / 4,
                                        NumElements: size / 4,
                                        StructureByteStride: 0,
                                        Flags: d3d12_ty::D3D12_BUFFER_SRV_FLAG_RAW,
                                    }
                                };
                                unsafe {
                                    self.raw.CreateShaderResourceView(
                                        data.buffer.resource.as_mut_ptr(),
                                        &raw_desc,
                                        handle,
                                    )
                                };
                            }
                            wgt::BufferBindingType::Storage { read_only: false } => {
                                let mut raw_desc = d3d12_ty::D3D12_UNORDERED_ACCESS_VIEW_DESC {
                                    Format: dxgiformat::DXGI_FORMAT_R32_TYPELESS,
                                    ViewDimension: d3d12_ty::D3D12_UAV_DIMENSION_BUFFER,
                                    u: unsafe { mem::zeroed() },
                                };
                                unsafe {
                                    *raw_desc.u.Buffer_mut() = d3d12_ty::D3D12_BUFFER_UAV {
                                        FirstElement: data.offset / 4,
                                        NumElements: size / 4,
                                        StructureByteStride: 0,
                                        CounterOffsetInBytes: 0,
                                        Flags: d3d12_ty::D3D12_BUFFER_UAV_FLAG_RAW,
                                    }
                                };
                                unsafe {
                                    self.raw.CreateUnorderedAccessView(
                                        data.buffer.resource.as_mut_ptr(),
                                        ptr::null_mut(),
                                        &raw_desc,
                                        handle,
                                    )
                                };
                            }
                        }
                        inner.stage.push(handle);
                    }
                }
                wgt::BindingType::Texture { .. } => {
                    let start = entry.resource_index as usize;
                    let end = start + entry.count as usize;
                    for data in &desc.textures[start..end] {
                        let handle = data.view.handle_srv.unwrap();
                        cpu_views.as_mut().unwrap().stage.push(handle.raw);
                    }
                }
                wgt::BindingType::StorageTexture { .. } => {
                    let start = entry.resource_index as usize;
                    let end = start + entry.count as usize;
                    for data in &desc.textures[start..end] {
                        let handle = data.view.handle_uav.unwrap();
                        cpu_views.as_mut().unwrap().stage.push(handle.raw);
                    }
                }
                wgt::BindingType::Sampler { .. } => {
                    let start = entry.resource_index as usize;
                    let end = start + entry.count as usize;
                    for data in &desc.samplers[start..end] {
                        cpu_samplers.as_mut().unwrap().stage.push(data.handle.raw);
                    }
                }
            }
        }

        let handle_views = match cpu_views {
            Some(inner) => {
                let dual = unsafe {
                    descriptor::upload(
                        self.raw.clone(),
                        &inner,
                        &self.shared.heap_views,
                        &desc.layout.copy_counts,
                    )
                }?;
                Some(dual)
            }
            None => None,
        };
        let handle_samplers = match cpu_samplers {
            Some(inner) => {
                let dual = unsafe {
                    descriptor::upload(
                        self.raw.clone(),
                        &inner,
                        &self.shared.heap_samplers,
                        &desc.layout.copy_counts,
                    )
                }?;
                Some(dual)
            }
            None => None,
        };

        Ok(super::BindGroup {
            handle_views,
            handle_samplers,
            dynamic_buffers,
        })
    }
    unsafe fn destroy_bind_group(&self, group: super::BindGroup) {
        if let Some(dual) = group.handle_views {
            self.shared.heap_views.free_slice(dual);
        }
        if let Some(dual) = group.handle_samplers {
            self.shared.heap_samplers.free_slice(dual);
        }
    }

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        let raw_name = desc.label.and_then(|label| ffi::CString::new(label).ok());
        match shader {
            crate::ShaderInput::Naga(naga) => Ok(super::ShaderModule { naga, raw_name }),
            crate::ShaderInput::SpirV(_) => {
                panic!("SPIRV_SHADER_PASSTHROUGH is not enabled for this backend")
            }
        }
    }
    unsafe fn destroy_shader_module(&self, _module: super::ShaderModule) {
        // just drop
    }

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        let (topology_class, topology) = conv::map_topology(desc.primitive.topology);
        let mut shader_stages = wgt::ShaderStages::VERTEX;

        let blob_vs =
            self.load_shader(&desc.vertex_stage, desc.layout, naga::ShaderStage::Vertex)?;
        let blob_fs = match desc.fragment_stage {
            Some(ref stage) => {
                shader_stages |= wgt::ShaderStages::FRAGMENT;
                Some(self.load_shader(stage, desc.layout, naga::ShaderStage::Fragment)?)
            }
            None => None,
        };

        let mut vertex_strides = [None; crate::MAX_VERTEX_BUFFERS];
        let mut input_element_descs = Vec::new();
        for (i, (stride, vbuf)) in vertex_strides
            .iter_mut()
            .zip(desc.vertex_buffers)
            .enumerate()
        {
            *stride = NonZeroU32::new(vbuf.array_stride as u32);
            let (slot_class, step_rate) = match vbuf.step_mode {
                wgt::VertexStepMode::Vertex => {
                    (d3d12_ty::D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0)
                }
                wgt::VertexStepMode::Instance => {
                    (d3d12_ty::D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1)
                }
            };
            for attribute in vbuf.attributes {
                input_element_descs.push(d3d12_ty::D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: NAGA_LOCATION_SEMANTIC.as_ptr() as *const _,
                    SemanticIndex: attribute.shader_location,
                    Format: auxil::dxgi::conv::map_vertex_format(attribute.format),
                    InputSlot: i as u32,
                    AlignedByteOffset: attribute.offset as u32,
                    InputSlotClass: slot_class,
                    InstanceDataStepRate: step_rate,
                });
            }
        }

        let mut rtv_formats = [dxgiformat::DXGI_FORMAT_UNKNOWN;
            d3d12_ty::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize];
        for (rtv_format, ct) in rtv_formats.iter_mut().zip(desc.color_targets) {
            if let Some(ct) = ct.as_ref() {
                *rtv_format = auxil::dxgi::conv::map_texture_format(ct.format);
            }
        }

        let bias = desc
            .depth_stencil
            .as_ref()
            .map(|ds| ds.bias)
            .unwrap_or_default();

        let raw_rasterizer = d3d12_ty::D3D12_RASTERIZER_DESC {
            FillMode: conv::map_polygon_mode(desc.primitive.polygon_mode),
            CullMode: match desc.primitive.cull_mode {
                None => d3d12_ty::D3D12_CULL_MODE_NONE,
                Some(wgt::Face::Front) => d3d12_ty::D3D12_CULL_MODE_FRONT,
                Some(wgt::Face::Back) => d3d12_ty::D3D12_CULL_MODE_BACK,
            },
            FrontCounterClockwise: match desc.primitive.front_face {
                wgt::FrontFace::Cw => 0,
                wgt::FrontFace::Ccw => 1,
            },
            DepthBias: bias.constant,
            DepthBiasClamp: bias.clamp,
            SlopeScaledDepthBias: bias.slope_scale,
            DepthClipEnable: BOOL::from(!desc.primitive.unclipped_depth),
            MultisampleEnable: BOOL::from(desc.multisample.count > 1),
            ForcedSampleCount: 0,
            AntialiasedLineEnable: 0,
            ConservativeRaster: if desc.primitive.conservative {
                d3d12_ty::D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
            } else {
                d3d12_ty::D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
            },
        };

        let raw_desc = d3d12_ty::D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: desc.layout.shared.signature.as_mut_ptr(),
            VS: *blob_vs.create_native_shader(),
            PS: match blob_fs {
                Some(ref shader) => *shader.create_native_shader(),
                None => *d3d12::Shader::null(),
            },
            GS: *d3d12::Shader::null(),
            DS: *d3d12::Shader::null(),
            HS: *d3d12::Shader::null(),
            StreamOutput: d3d12_ty::D3D12_STREAM_OUTPUT_DESC {
                pSODeclaration: ptr::null(),
                NumEntries: 0,
                pBufferStrides: ptr::null(),
                NumStrides: 0,
                RasterizedStream: 0,
            },
            BlendState: d3d12_ty::D3D12_BLEND_DESC {
                AlphaToCoverageEnable: BOOL::from(desc.multisample.alpha_to_coverage_enabled),
                IndependentBlendEnable: 1,
                RenderTarget: conv::map_render_targets(desc.color_targets),
            },
            SampleMask: desc.multisample.mask as u32,
            RasterizerState: raw_rasterizer,
            DepthStencilState: match desc.depth_stencil {
                Some(ref ds) => conv::map_depth_stencil(ds),
                None => unsafe { mem::zeroed() },
            },
            InputLayout: d3d12_ty::D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: if input_element_descs.is_empty() {
                    ptr::null()
                } else {
                    input_element_descs.as_ptr()
                },
                NumElements: input_element_descs.len() as u32,
            },
            IBStripCutValue: match desc.primitive.strip_index_format {
                Some(wgt::IndexFormat::Uint16) => {
                    d3d12_ty::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
                }
                Some(wgt::IndexFormat::Uint32) => {
                    d3d12_ty::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF
                }
                None => d3d12_ty::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            },
            PrimitiveTopologyType: topology_class,
            NumRenderTargets: desc.color_targets.len() as u32,
            RTVFormats: rtv_formats,
            DSVFormat: desc
                .depth_stencil
                .as_ref()
                .map_or(dxgiformat::DXGI_FORMAT_UNKNOWN, |ds| {
                    auxil::dxgi::conv::map_texture_format(ds.format)
                }),
            SampleDesc: dxgitype::DXGI_SAMPLE_DESC {
                Count: desc.multisample.count,
                Quality: 0,
            },
            NodeMask: 0,
            CachedPSO: d3d12_ty::D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: ptr::null(),
                CachedBlobSizeInBytes: 0,
            },
            Flags: d3d12_ty::D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        let mut raw = d3d12::PipelineState::null();
        let hr = {
            profiling::scope!("ID3D12Device::CreateGraphicsPipelineState");
            unsafe {
                self.raw.CreateGraphicsPipelineState(
                    &raw_desc,
                    &d3d12_ty::ID3D12PipelineState::uuidof(),
                    raw.mut_void(),
                )
            }
        };

        unsafe { blob_vs.destroy() };
        if let Some(blob_fs) = blob_fs {
            unsafe { blob_fs.destroy() };
        };

        hr.into_result()
            .map_err(|err| crate::PipelineError::Linkage(shader_stages, err.into_owned()))?;

        if let Some(name) = desc.label {
            let cwstr = conv::map_label(name);
            unsafe { raw.SetName(cwstr.as_ptr()) };
        }

        Ok(super::RenderPipeline {
            raw,
            layout: desc.layout.shared.clone(),
            topology,
            vertex_strides,
        })
    }
    unsafe fn destroy_render_pipeline(&self, _pipeline: super::RenderPipeline) {}

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        let blob_cs = self.load_shader(&desc.stage, desc.layout, naga::ShaderStage::Compute)?;

        let pair = {
            profiling::scope!("ID3D12Device::CreateComputePipelineState");
            self.raw.create_compute_pipeline_state(
                &desc.layout.shared.signature,
                blob_cs.create_native_shader(),
                0,
                d3d12::CachedPSO::null(),
                d3d12::PipelineStateFlags::empty(),
            )
        };

        unsafe { blob_cs.destroy() };

        let raw = pair.into_result().map_err(|err| {
            crate::PipelineError::Linkage(wgt::ShaderStages::COMPUTE, err.into_owned())
        })?;

        if let Some(name) = desc.label {
            let cwstr = conv::map_label(name);
            unsafe { raw.SetName(cwstr.as_ptr()) };
        }

        Ok(super::ComputePipeline {
            raw,
            layout: desc.layout.shared.clone(),
        })
    }
    unsafe fn destroy_compute_pipeline(&self, _pipeline: super::ComputePipeline) {}

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<super::QuerySet, crate::DeviceError> {
        let (heap_ty, raw_ty) = match desc.ty {
            wgt::QueryType::Occlusion => (
                d3d12::QueryHeapType::Occlusion,
                d3d12_ty::D3D12_QUERY_TYPE_BINARY_OCCLUSION,
            ),
            wgt::QueryType::PipelineStatistics(_) => (
                d3d12::QueryHeapType::PipelineStatistics,
                d3d12_ty::D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            ),
            wgt::QueryType::Timestamp => (
                d3d12::QueryHeapType::Timestamp,
                d3d12_ty::D3D12_QUERY_TYPE_TIMESTAMP,
            ),
        };

        let raw = self
            .raw
            .create_query_heap(heap_ty, desc.count, 0)
            .into_device_result("Query heap creation")?;

        if let Some(label) = desc.label {
            let cwstr = conv::map_label(label);
            unsafe { raw.SetName(cwstr.as_ptr()) };
        }

        Ok(super::QuerySet { raw, raw_ty })
    }
    unsafe fn destroy_query_set(&self, _set: super::QuerySet) {}

    unsafe fn create_fence(&self) -> Result<super::Fence, crate::DeviceError> {
        let mut raw = d3d12::Fence::null();
        let hr = unsafe {
            self.raw.CreateFence(
                0,
                d3d12_ty::D3D12_FENCE_FLAG_NONE,
                &d3d12_ty::ID3D12Fence::uuidof(),
                raw.mut_void(),
            )
        };
        hr.into_device_result("Fence creation")?;
        Ok(super::Fence { raw })
    }
    unsafe fn destroy_fence(&self, _fence: super::Fence) {}
    unsafe fn get_fence_value(
        &self,
        fence: &super::Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        Ok(unsafe { fence.raw.GetCompletedValue() })
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        if unsafe { fence.raw.GetCompletedValue() } >= value {
            return Ok(true);
        }
        let hr = fence.raw.set_event_on_completion(self.idler.event, value);
        hr.into_device_result("Set event")?;

        match unsafe { synchapi::WaitForSingleObject(self.idler.event.0, timeout_ms) } {
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
        #[cfg(feature = "renderdoc")]
        {
            unsafe {
                self.render_doc
                    .start_frame_capture(self.raw.as_mut_ptr() as *mut _, ptr::null_mut())
            }
        }
        #[cfg(not(feature = "renderdoc"))]
        false
    }

    unsafe fn stop_capture(&self) {
        #[cfg(feature = "renderdoc")]
        unsafe {
            self.render_doc
                .end_frame_capture(self.raw.as_mut_ptr() as *mut _, ptr::null_mut())
        }
    }
}
