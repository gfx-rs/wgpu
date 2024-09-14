use std::{
    ffi,
    mem::{self, size_of},
    num::NonZeroU32,
    ptr,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use windows::{
    core::Interface as _,
    Win32::{
        Foundation,
        Graphics::{Direct3D12, Dxgi},
        System::Threading,
    },
};

use super::{conv, descriptor, D3D12Lib};
use crate::{
    auxil::{self, dxgi::result::HResult},
    dx12::{borrow_optional_interface_temporarily, shader_compilation, Event},
};

// this has to match Naga's HLSL backend, and also needs to be null-terminated
const NAGA_LOCATION_SEMANTIC: &[u8] = b"LOC\0";

impl super::Device {
    pub(super) fn new(
        raw: Direct3D12::ID3D12Device,
        present_queue: Direct3D12::ID3D12CommandQueue,
        limits: &wgt::Limits,
        memory_hints: &wgt::MemoryHints,
        private_caps: super::PrivateCapabilities,
        library: &Arc<D3D12Lib>,
        dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
    ) -> Result<Self, crate::DeviceError> {
        let mem_allocator = super::suballocation::create_allocator_wrapper(&raw, memory_hints)?;

        let idle_fence: Direct3D12::ID3D12Fence = unsafe {
            profiling::scope!("ID3D12Device::CreateFence");
            raw.CreateFence(0, Direct3D12::D3D12_FENCE_FLAG_NONE)
        }
        .into_device_result("Idle fence creation")?;

        let raw_desc = Direct3D12::D3D12_RESOURCE_DESC {
            Dimension: Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: super::ZERO_BUFFER_SIZE,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: Direct3D12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: Direct3D12::D3D12_RESOURCE_FLAG_NONE,
        };

        let heap_properties = Direct3D12::D3D12_HEAP_PROPERTIES {
            Type: Direct3D12::D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: Direct3D12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: match private_caps.memory_architecture {
                super::MemoryArchitecture::Unified { .. } => Direct3D12::D3D12_MEMORY_POOL_L0,
                super::MemoryArchitecture::NonUnified => Direct3D12::D3D12_MEMORY_POOL_L1,
            },
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        profiling::scope!("Zero Buffer Allocation");
        let mut zero_buffer = None::<Direct3D12::ID3D12Resource>;
        unsafe {
            raw.CreateCommittedResource(
                &heap_properties,
                Direct3D12::D3D12_HEAP_FLAG_NONE,
                &raw_desc,
                Direct3D12::D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut zero_buffer,
            )
        }
        .into_device_result("Zero buffer creation")?;

        let zero_buffer = zero_buffer.ok_or(crate::DeviceError::Unexpected)?;

        // Note: without `D3D12_HEAP_FLAG_CREATE_NOT_ZEROED`
        // this resource is zeroed by default.

        // maximum number of CBV/SRV/UAV descriptors in heap for Tier 1
        let capacity_views = limits.max_non_sampler_bindings as u64;
        let capacity_samplers = 2_048;

        fn create_command_signature(
            raw: &Direct3D12::ID3D12Device,
            byte_stride: usize,
            arguments: &[Direct3D12::D3D12_INDIRECT_ARGUMENT_DESC],
            node_mask: u32,
        ) -> Result<Direct3D12::ID3D12CommandSignature, crate::DeviceError> {
            let mut signature = None;
            unsafe {
                raw.CreateCommandSignature(
                    &Direct3D12::D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: byte_stride as u32,
                        NumArgumentDescs: arguments.len() as u32,
                        pArgumentDescs: arguments.as_ptr(),
                        NodeMask: node_mask,
                    },
                    None,
                    &mut signature,
                )
            }
            .into_device_result("Command signature creation")?;
            signature.ok_or(crate::DeviceError::Unexpected)
        }

        let shared = super::DeviceShared {
            zero_buffer,
            cmd_signatures: super::CommandSignatures {
                draw: create_command_signature(
                    &raw,
                    size_of::<wgt::DrawIndirectArgs>(),
                    &[Direct3D12::D3D12_INDIRECT_ARGUMENT_DESC {
                        Type: Direct3D12::D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                        ..Default::default()
                    }],
                    0,
                )?,
                draw_indexed: create_command_signature(
                    &raw,
                    size_of::<wgt::DrawIndexedIndirectArgs>(),
                    &[Direct3D12::D3D12_INDIRECT_ARGUMENT_DESC {
                        Type: Direct3D12::D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
                        ..Default::default()
                    }],
                    0,
                )?,
                dispatch: create_command_signature(
                    &raw,
                    size_of::<wgt::DispatchIndirectArgs>(),
                    &[Direct3D12::D3D12_INDIRECT_ARGUMENT_DESC {
                        Type: Direct3D12::D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                        ..Default::default()
                    }],
                    0,
                )?,
            },
            heap_views: descriptor::GeneralHeap::new(
                &raw,
                Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                capacity_views,
            )?,
            heap_samplers: descriptor::GeneralHeap::new(
                &raw,
                Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                capacity_samplers,
            )?,
        };

        let mut rtv_pool =
            descriptor::CpuPool::new(raw.clone(), Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let null_rtv_handle = rtv_pool.alloc_handle()?;
        // A null pResource is used to initialize a null descriptor,
        // which guarantees D3D11-like null binding behavior (reading 0s, writes are discarded)
        unsafe {
            raw.CreateRenderTargetView(
                None,
                Some(&Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                    ViewDimension: Direct3D12::D3D12_RTV_DIMENSION_TEXTURE2D,
                    Anonymous: Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC_0 {
                        Texture2D: Direct3D12::D3D12_TEX2D_RTV {
                            MipSlice: 0,
                            PlaneSlice: 0,
                        },
                    },
                }),
                null_rtv_handle.raw,
            )
        };

        Ok(super::Device {
            raw: raw.clone(),
            present_queue,
            idler: super::Idler {
                fence: idle_fence,
                event: Event::create(false, false)?,
            },
            private_caps,
            shared: Arc::new(shared),
            rtv_pool: Mutex::new(rtv_pool),
            dsv_pool: Mutex::new(descriptor::CpuPool::new(
                raw.clone(),
                Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
            )),
            srv_uav_pool: Mutex::new(descriptor::CpuPool::new(
                raw.clone(),
                Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            )),
            sampler_pool: Mutex::new(descriptor::CpuPool::new(
                raw,
                Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            )),
            library: Arc::clone(library),
            #[cfg(feature = "renderdoc")]
            render_doc: Default::default(),
            null_rtv_handle,
            mem_allocator,
            dxc_container,
            counters: Default::default(),
        })
    }

    // Blocks until the dedicated present queue is finished with all of its work.
    //
    // Once this method completes, the surface is able to be resized or deleted.
    pub(super) unsafe fn wait_for_present_queue_idle(&self) -> Result<(), crate::DeviceError> {
        let cur_value = unsafe { self.idler.fence.GetCompletedValue() };
        if cur_value == !0 {
            return Err(crate::DeviceError::Lost);
        }

        let value = cur_value + 1;
        unsafe { self.present_queue.Signal(&self.idler.fence, value) }
            .into_device_result("Signal")?;
        let hr = unsafe {
            self.idler
                .fence
                .SetEventOnCompletion(value, self.idler.event.0)
        };
        hr.into_device_result("Set event")?;
        unsafe { Threading::WaitForSingleObject(self.idler.event.0, Threading::INFINITE) };
        Ok(())
    }

    /// When generating the vertex shader, the fragment stage must be passed if it exists!
    /// Otherwise, the generated HLSL may be incorrect since the fragment shader inputs are
    /// allowed to be a subset of the vertex outputs.
    fn load_shader(
        &self,
        stage: &crate::ProgrammableStage<super::ShaderModule>,
        layout: &super::PipelineLayout,
        naga_stage: naga::ShaderStage,
        fragment_stage: Option<&crate::ProgrammableStage<super::ShaderModule>>,
    ) -> Result<super::CompiledShader, crate::PipelineError> {
        use naga::back::hlsl;

        let frag_ep = fragment_stage
            .map(|fs_stage| {
                hlsl::FragmentEntryPoint::new(&fs_stage.module.naga.module, fs_stage.entry_point)
                    .ok_or(crate::PipelineError::EntryPoint(
                        naga::ShaderStage::Fragment,
                    ))
            })
            .transpose()?;

        let stage_bit = auxil::map_naga_stage(naga_stage);

        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &stage.module.naga.module,
            &stage.module.naga.info,
            stage.constants,
        )
        .map_err(|e| crate::PipelineError::PipelineConstants(stage_bit, format!("HLSL: {e:?}")))?;

        let needs_temp_options = stage.zero_initialize_workgroup_memory
            != layout.naga_options.zero_initialize_workgroup_memory;
        let mut temp_options;
        let naga_options = if needs_temp_options {
            temp_options = layout.naga_options.clone();
            temp_options.zero_initialize_workgroup_memory = stage.zero_initialize_workgroup_memory;
            &temp_options
        } else {
            &layout.naga_options
        };

        //TODO: reuse the writer
        let mut source = String::new();
        let mut writer = hlsl::Writer::new(&mut source, naga_options);
        let reflection_info = {
            profiling::scope!("naga::back::hlsl::write");
            writer
                .write(&module, &info, frag_ep.as_ref())
                .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("HLSL: {e:?}")))?
        };

        let full_stage = format!(
            "{}_{}",
            naga_stage.to_hlsl_str(),
            naga_options.shader_model.to_str()
        );

        let ep_index = module
            .entry_points
            .iter()
            .position(|ep| ep.stage == naga_stage && ep.name == stage.entry_point)
            .ok_or(crate::PipelineError::EntryPoint(naga_stage))?;

        let raw_ep = reflection_info.entry_point_names[ep_index]
            .as_ref()
            .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("{e}")))?;

        let source_name = stage.module.raw_name.as_deref();

        // Compile with DXC if available, otherwise fall back to FXC
        let result = if let Some(ref dxc_container) = self.dxc_container {
            shader_compilation::compile_dxc(
                self,
                &source,
                source_name,
                raw_ep,
                stage_bit,
                &full_stage,
                dxc_container,
            )
        } else {
            shader_compilation::compile_fxc(
                self,
                &source,
                source_name,
                raw_ep,
                stage_bit,
                &full_stage,
            )
        };

        let log_level = if result.is_ok() {
            log::Level::Info
        } else {
            log::Level::Error
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

    pub fn raw_device(&self) -> &Direct3D12::ID3D12Device {
        &self.raw
    }

    pub fn raw_queue(&self) -> &Direct3D12::ID3D12CommandQueue {
        &self.present_queue
    }

    pub unsafe fn texture_from_raw(
        resource: Direct3D12::ID3D12Resource,
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
        resource: Direct3D12::ID3D12Resource,
        size: wgt::BufferAddress,
    ) -> super::Buffer {
        super::Buffer {
            resource,
            size,
            allocation: None,
        }
    }
}

impl crate::Device for super::Device {
    type A = super::Api;

    unsafe fn exit(self, _queue: super::Queue) {
        self.rtv_pool.lock().free_handle(self.null_rtv_handle);
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let mut size = desc.size;
        if desc.usage.contains(crate::BufferUses::UNIFORM) {
            let align_mask = Direct3D12::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as u64 - 1;
            size = ((size - 1) | align_mask) + 1;
        }

        let raw_desc = Direct3D12::D3D12_RESOURCE_DESC {
            Dimension: Direct3D12::D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: Direct3D12::D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: conv::map_buffer_usage_to_resource_flags(desc.usage),
        };

        let (resource, allocation) =
            super::suballocation::create_buffer_resource(self, desc, raw_desc)?;

        if let Some(label) = desc.label {
            unsafe { resource.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.buffers.add(1);

        Ok(super::Buffer {
            resource,
            size,
            allocation,
        })
    }

    unsafe fn destroy_buffer(&self, mut buffer: super::Buffer) {
        // Always Some except on Intel Xe: https://github.com/gfx-rs/wgpu/issues/3552
        if let Some(alloc) = buffer.allocation.take() {
            // Resource should be dropped before free suballocation
            drop(buffer);

            super::suballocation::free_buffer_allocation(self, alloc, &self.mem_allocator);
        }

        self.counters.buffers.sub(1);
    }

    unsafe fn add_raw_buffer(&self, _buffer: &super::Buffer) {
        self.counters.buffers.add(1);
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let mut ptr = ptr::null_mut();
        // TODO: 0 for subresource should be fine here until map and unmap buffer is subresource aware?
        unsafe { buffer.resource.Map(0, None, Some(&mut ptr)) }.into_device_result("Map buffer")?;

        Ok(crate::BufferMapping {
            ptr: ptr::NonNull::new(unsafe { ptr.offset(range.start as isize).cast::<u8>() })
                .unwrap(),
            //TODO: double-check this. Documentation is a bit misleading -
            // it implies that Map/Unmap is needed to invalidate/flush memory.
            is_coherent: true,
        })
    }

    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) {
        unsafe { buffer.resource.Unmap(0, None) };
    }

    unsafe fn flush_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let raw_desc = Direct3D12::D3D12_RESOURCE_DESC {
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
            SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: desc.sample_count,
                Quality: 0,
            },
            Layout: Direct3D12::D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: conv::map_texture_usage_to_resource_flags(desc.usage),
        };

        let (resource, allocation) =
            super::suballocation::create_texture_resource(self, desc, raw_desc)?;

        if let Some(label) = desc.label {
            unsafe { resource.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.textures.add(1);

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
            // Resource should be dropped before free suballocation
            drop(texture);

            super::suballocation::free_texture_allocation(
                self,
                alloc,
                // SAFETY: for allocations to exist, the allocator must exist
                &self.mem_allocator,
            );
        }

        self.counters.textures.sub(1);
    }

    unsafe fn add_raw_texture(&self, _texture: &super::Texture) {
        self.counters.textures.add(1);
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        let view_desc = desc.to_internal(texture);

        self.counters.texture_views.add(1);

        Ok(super::TextureView {
            raw_format: view_desc.rtv_dsv_format,
            aspects: view_desc.aspects,
            target_base: (
                texture.resource.clone(),
                texture.calc_subresource(desc.range.base_mip_level, desc.range.base_array_layer, 0),
            ),
            handle_srv: if desc.usage.intersects(crate::TextureUses::RESOURCE) {
                match unsafe { view_desc.to_srv() } {
                    Some(raw_desc) => {
                        let handle = self.srv_uav_pool.lock().alloc_handle()?;
                        unsafe {
                            self.raw.CreateShaderResourceView(
                                &texture.resource,
                                Some(&raw_desc),
                                handle.raw,
                            )
                        };
                        Some(handle)
                    }
                    None => None,
                }
            } else {
                None
            },
            handle_uav: if desc.usage.intersects(
                crate::TextureUses::STORAGE_READ | crate::TextureUses::STORAGE_READ_WRITE,
            ) {
                match unsafe { view_desc.to_uav() } {
                    Some(raw_desc) => {
                        let handle = self.srv_uav_pool.lock().alloc_handle()?;
                        unsafe {
                            self.raw.CreateUnorderedAccessView(
                                &texture.resource,
                                None,
                                Some(&raw_desc),
                                handle.raw,
                            );
                        }
                        Some(handle)
                    }
                    None => None,
                }
            } else {
                None
            },
            handle_rtv: if desc.usage.intersects(crate::TextureUses::COLOR_TARGET) {
                let raw_desc = unsafe { view_desc.to_rtv() };
                let handle = self.rtv_pool.lock().alloc_handle()?;
                unsafe {
                    self.raw
                        .CreateRenderTargetView(&texture.resource, Some(&raw_desc), handle.raw)
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
                let handle = self.dsv_pool.lock().alloc_handle()?;
                unsafe {
                    self.raw
                        .CreateDepthStencilView(&texture.resource, Some(&raw_desc), handle.raw)
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
                let handle = self.dsv_pool.lock().alloc_handle()?;
                unsafe {
                    self.raw
                        .CreateDepthStencilView(&texture.resource, Some(&raw_desc), handle.raw)
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

        self.counters.texture_views.sub(1);
    }

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        let handle = self.sampler_pool.lock().alloc_handle()?;

        let reduction = match desc.compare {
            Some(_) => Direct3D12::D3D12_FILTER_REDUCTION_TYPE_COMPARISON,
            None => Direct3D12::D3D12_FILTER_REDUCTION_TYPE_STANDARD,
        };
        let mut filter = Direct3D12::D3D12_FILTER(
            conv::map_filter_mode(desc.min_filter).0 << Direct3D12::D3D12_MIN_FILTER_SHIFT
                | conv::map_filter_mode(desc.mag_filter).0 << Direct3D12::D3D12_MAG_FILTER_SHIFT
                | conv::map_filter_mode(desc.mipmap_filter).0 << Direct3D12::D3D12_MIP_FILTER_SHIFT
                | reduction.0 << Direct3D12::D3D12_FILTER_REDUCTION_TYPE_SHIFT,
        );

        if desc.anisotropy_clamp != 1 {
            filter.0 |= Direct3D12::D3D12_FILTER_ANISOTROPIC.0;
        };

        let border_color = conv::map_border_color(desc.border_color);

        unsafe {
            self.raw.CreateSampler(
                &Direct3D12::D3D12_SAMPLER_DESC {
                    Filter: filter,
                    AddressU: conv::map_address_mode(desc.address_modes[0]),
                    AddressV: conv::map_address_mode(desc.address_modes[1]),
                    AddressW: conv::map_address_mode(desc.address_modes[2]),
                    MipLODBias: 0f32,
                    MaxAnisotropy: desc.anisotropy_clamp as u32,

                    ComparisonFunc: conv::map_comparison(
                        desc.compare.unwrap_or(wgt::CompareFunction::Always),
                    ),
                    BorderColor: border_color,
                    MinLOD: desc.lod_clamp.start,
                    MaxLOD: desc.lod_clamp.end,
                },
                handle.raw,
            )
        };

        self.counters.samplers.add(1);

        Ok(super::Sampler { handle })
    }

    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        self.sampler_pool.lock().free_handle(sampler.handle);
        self.counters.samplers.sub(1);
    }

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Queue>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        let allocator: Direct3D12::ID3D12CommandAllocator = unsafe {
            self.raw
                .CreateCommandAllocator(Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .into_device_result("Command allocator creation")?;

        if let Some(label) = desc.label {
            unsafe { allocator.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.command_encoders.add(1);

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

    unsafe fn destroy_command_encoder(&self, _encoder: super::CommandEncoder) {
        self.counters.command_encoders.sub(1);
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
                wgt::BindingType::AccelerationStructure => todo!(),
            }
        }

        self.counters.bind_group_layouts.add(1);

        let num_views = num_buffer_views + num_texture_views;
        Ok(super::BindGroupLayout {
            entries: desc.entries.to_vec(),
            cpu_heap_views: if num_views != 0 {
                let heap = descriptor::CpuHeap::new(
                    &self.raw,
                    Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    num_views,
                )?;
                Some(heap)
            } else {
                None
            },
            cpu_heap_samplers: if num_samplers != 0 {
                let heap = descriptor::CpuHeap::new(
                    &self.raw,
                    Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    num_samplers,
                )?;
                Some(heap)
            } else {
                None
            },
            copy_counts: vec![1; num_views.max(num_samplers) as usize],
        })
    }

    unsafe fn destroy_bind_group_layout(&self, _bg_layout: super::BindGroupLayout) {
        self.counters.bind_group_layouts.sub(1);
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::BindGroupLayout>,
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

        //TODO: put lower bind group indices further down the root signature. See:
        // https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html#binding-model
        // Currently impossible because wgpu-core only re-binds the descriptor sets based
        // on Vulkan-like layout compatibility rules.

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
            parameters.push(Direct3D12::D3D12_ROOT_PARAMETER {
                ParameterType: Direct3D12::D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: Direct3D12::D3D12_ROOT_PARAMETER_0 {
                    Constants: Direct3D12::D3D12_ROOT_CONSTANTS {
                        ShaderRegister: bind_cbv.register,
                        RegisterSpace: bind_cbv.space as u32,
                        Num32BitValues: size,
                    },
                },
                ShaderVisibility: Direct3D12::D3D12_SHADER_VISIBILITY_ALL,
            });
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
                    Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_CBV => &mut bind_cbv,
                    Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_SRV => &mut bind_srv,
                    Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_UAV => &mut bind_uav,
                    Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER => continue,
                    _ => todo!(),
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
                ranges.push(Direct3D12::D3D12_DESCRIPTOR_RANGE {
                    RangeType: range_ty,
                    NumDescriptors: entry.count.map_or(1, |count| count.get()),
                    BaseShaderRegister: bt.register,
                    RegisterSpace: bt.space as u32,
                    OffsetInDescriptorsFromTableStart:
                        Direct3D12::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                });
                bt.register += entry.count.map(NonZeroU32::get).unwrap_or(1);
            }
            if ranges.len() > range_base {
                let range = &ranges[range_base..];
                parameters.push(Direct3D12::D3D12_ROOT_PARAMETER {
                    ParameterType: Direct3D12::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: Direct3D12::D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: Direct3D12::D3D12_ROOT_DESCRIPTOR_TABLE {
                            NumDescriptorRanges: range.len() as u32,
                            pDescriptorRanges: range.as_ptr(),
                        },
                    },
                    ShaderVisibility: conv::map_visibility(visibility_view_static),
                });
                info.tables |= super::TableTypes::SRV_CBV_UAV;
            }

            // Sampler descriptor tables
            range_base = ranges.len();
            for entry in bgl.entries.iter() {
                let range_ty = match entry.ty {
                    wgt::BindingType::Sampler { .. } => {
                        Direct3D12::D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER
                    }
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
                ranges.push(Direct3D12::D3D12_DESCRIPTOR_RANGE {
                    RangeType: range_ty,
                    NumDescriptors: entry.count.map_or(1, |count| count.get()),
                    BaseShaderRegister: bind_sampler.register,
                    RegisterSpace: bind_sampler.space as u32,
                    OffsetInDescriptorsFromTableStart:
                        Direct3D12::D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                });
                bind_sampler.register += entry.count.map(NonZeroU32::get).unwrap_or(1);
            }
            if ranges.len() > range_base {
                let range = &ranges[range_base..];
                parameters.push(Direct3D12::D3D12_ROOT_PARAMETER {
                    ParameterType: Direct3D12::D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: Direct3D12::D3D12_ROOT_PARAMETER_0 {
                        DescriptorTable: Direct3D12::D3D12_ROOT_DESCRIPTOR_TABLE {
                            NumDescriptorRanges: range.len() as u32,
                            pDescriptorRanges: range.as_ptr(),
                        },
                    },
                    ShaderVisibility: conv::map_visibility(visibility_sampler),
                });
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
                        Direct3D12::D3D12_ROOT_PARAMETER_TYPE_CBV,
                        &mut bind_cbv,
                    ),
                    wgt::BufferBindingType::Storage { read_only: true } => (
                        super::BufferViewKind::ShaderResource,
                        Direct3D12::D3D12_ROOT_PARAMETER_TYPE_SRV,
                        &mut bind_srv,
                    ),
                    wgt::BufferBindingType::Storage { read_only: false } => (
                        super::BufferViewKind::UnorderedAccess,
                        Direct3D12::D3D12_ROOT_PARAMETER_TYPE_UAV,
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

                parameters.push(Direct3D12::D3D12_ROOT_PARAMETER {
                    ParameterType: parameter_ty,
                    Anonymous: Direct3D12::D3D12_ROOT_PARAMETER_0 {
                        Descriptor: Direct3D12::D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: bt.register,
                            RegisterSpace: bt.space as u32,
                        },
                    },
                    ShaderVisibility: dynamic_buffers_visibility,
                });

                bt.register += entry.count.map_or(1, NonZeroU32::get);
            }

            bind_group_infos.push(info);
        }

        // Ensure that we didn't reallocate!
        debug_assert_eq!(ranges.len(), total_non_dynamic_entries);

        let (special_constants_root_index, special_constants_binding) = if desc.flags.intersects(
            crate::PipelineLayoutFlags::FIRST_VERTEX_INSTANCE
                | crate::PipelineLayoutFlags::NUM_WORK_GROUPS,
        ) {
            let parameter_index = parameters.len();
            parameters.push(Direct3D12::D3D12_ROOT_PARAMETER {
                ParameterType: Direct3D12::D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: Direct3D12::D3D12_ROOT_PARAMETER_0 {
                    Constants: Direct3D12::D3D12_ROOT_CONSTANTS {
                        ShaderRegister: bind_cbv.register,
                        RegisterSpace: bind_cbv.space as u32,
                        Num32BitValues: 3, // 0 = first_vertex, 1 = first_instance, 2 = other
                    },
                },
                ShaderVisibility: Direct3D12::D3D12_SHADER_VISIBILITY_ALL, // really needed for VS and CS only,
            });
            let binding = bind_cbv.clone();
            bind_cbv.register += 1;
            (Some(parameter_index as u32), Some(binding))
        } else {
            (None, None)
        };

        let blob = self.library.serialize_root_signature(
            Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_0,
            &parameters,
            &[],
            Direct3D12::D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        )?;

        let raw = unsafe {
            self.raw
                .CreateRootSignature::<Direct3D12::ID3D12RootSignature>(0, blob.as_slice())
        }
        .into_device_result("Root signature creation")?;

        if let Some(label) = desc.label {
            unsafe { raw.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.pipeline_layouts.add(1);

        Ok(super::PipelineLayout {
            shared: super::PipelineLayoutShared {
                signature: Some(raw),
                total_root_elements: parameters.len() as super::RootIndex,
                special_constants_root_index,
                root_constant_info,
            },
            bind_group_infos,
            naga_options: hlsl::Options {
                shader_model: self.private_caps.shader_model,
                binding_map,
                fake_missing_bindings: false,
                special_constants_binding,
                push_constants_target,
                zero_initialize_workgroup_memory: true,
            },
        })
    }

    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: super::PipelineLayout) {
        self.counters.pipeline_layouts.sub(1);
    }

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<
            super::BindGroupLayout,
            super::Buffer,
            super::Sampler,
            super::TextureView,
            super::AccelerationStructure,
        >,
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

        let layout_and_entry_iter = desc.entries.iter().map(|entry| {
            let layout = desc
                .layout
                .entries
                .iter()
                .find(|layout_entry| layout_entry.binding == entry.binding)
                .expect("internal error: no layout entry found with binding slot");
            (layout, entry)
        });
        for (layout, entry) in layout_and_entry_iter {
            match layout.ty {
                wgt::BindingType::Buffer {
                    has_dynamic_offset: true,
                    ..
                } => {
                    let start = entry.resource_index as usize;
                    let end = start + entry.count as usize;
                    for data in &desc.buffers[start..end] {
                        dynamic_buffers.push(Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: data.resolve_address(),
                        });
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
                                    Direct3D12::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1;
                                let raw_desc = Direct3D12::D3D12_CONSTANT_BUFFER_VIEW_DESC {
                                    BufferLocation: gpu_address,
                                    SizeInBytes: ((size - 1) | size_mask) + 1,
                                };
                                unsafe {
                                    self.raw.CreateConstantBufferView(Some(&raw_desc), handle)
                                };
                            }
                            wgt::BufferBindingType::Storage { read_only: true } => {
                                let raw_desc = Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC {
                                    Format: Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                                    Shader4ComponentMapping:
                                        Direct3D12::D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                                    ViewDimension: Direct3D12::D3D12_SRV_DIMENSION_BUFFER,
                                    Anonymous: Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                        Buffer: Direct3D12::D3D12_BUFFER_SRV {
                                            FirstElement: data.offset / 4,
                                            NumElements: size / 4,
                                            StructureByteStride: 0,
                                            Flags: Direct3D12::D3D12_BUFFER_SRV_FLAG_RAW,
                                        },
                                    },
                                };
                                unsafe {
                                    self.raw.CreateShaderResourceView(
                                        &data.buffer.resource,
                                        Some(&raw_desc),
                                        handle,
                                    )
                                };
                            }
                            wgt::BufferBindingType::Storage { read_only: false } => {
                                let raw_desc = Direct3D12::D3D12_UNORDERED_ACCESS_VIEW_DESC {
                                    Format: Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                                    ViewDimension: Direct3D12::D3D12_UAV_DIMENSION_BUFFER,
                                    Anonymous: Direct3D12::D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                        Buffer: Direct3D12::D3D12_BUFFER_UAV {
                                            FirstElement: data.offset / 4,
                                            NumElements: size / 4,
                                            StructureByteStride: 0,
                                            CounterOffsetInBytes: 0,
                                            Flags: Direct3D12::D3D12_BUFFER_UAV_FLAG_RAW,
                                        },
                                    },
                                };
                                unsafe {
                                    self.raw.CreateUnorderedAccessView(
                                        &data.buffer.resource,
                                        None,
                                        Some(&raw_desc),
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
                wgt::BindingType::AccelerationStructure => todo!(),
            }
        }

        let handle_views = match cpu_views {
            Some(inner) => {
                let dual = unsafe {
                    descriptor::upload(
                        &self.raw,
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
                        &self.raw,
                        &inner,
                        &self.shared.heap_samplers,
                        &desc.layout.copy_counts,
                    )
                }?;
                Some(dual)
            }
            None => None,
        };

        self.counters.bind_groups.add(1);

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

        self.counters.bind_groups.sub(1);
    }

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        self.counters.shader_modules.add(1);

        let raw_name = desc.label.and_then(|label| ffi::CString::new(label).ok());
        match shader {
            crate::ShaderInput::Naga(naga) => Ok(super::ShaderModule { naga, raw_name }),
            crate::ShaderInput::SpirV(_) => {
                panic!("SPIRV_SHADER_PASSTHROUGH is not enabled for this backend")
            }
        }
    }
    unsafe fn destroy_shader_module(&self, _module: super::ShaderModule) {
        self.counters.shader_modules.sub(1);
        // just drop
    }

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<
            super::PipelineLayout,
            super::ShaderModule,
            super::PipelineCache,
        >,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        let (topology_class, topology) = conv::map_topology(desc.primitive.topology);
        let mut shader_stages = wgt::ShaderStages::VERTEX;

        let blob_vs = self.load_shader(
            &desc.vertex_stage,
            desc.layout,
            naga::ShaderStage::Vertex,
            desc.fragment_stage.as_ref(),
        )?;
        let blob_fs = match desc.fragment_stage {
            Some(ref stage) => {
                shader_stages |= wgt::ShaderStages::FRAGMENT;
                Some(self.load_shader(stage, desc.layout, naga::ShaderStage::Fragment, None)?)
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
                    (Direct3D12::D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0)
                }
                wgt::VertexStepMode::Instance => {
                    (Direct3D12::D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1)
                }
            };
            for attribute in vbuf.attributes {
                input_element_descs.push(Direct3D12::D3D12_INPUT_ELEMENT_DESC {
                    SemanticName: windows::core::PCSTR(NAGA_LOCATION_SEMANTIC.as_ptr()),
                    SemanticIndex: attribute.shader_location,
                    Format: auxil::dxgi::conv::map_vertex_format(attribute.format),
                    InputSlot: i as u32,
                    AlignedByteOffset: attribute.offset as u32,
                    InputSlotClass: slot_class,
                    InstanceDataStepRate: step_rate,
                });
            }
        }

        let mut rtv_formats = [Dxgi::Common::DXGI_FORMAT_UNKNOWN;
            Direct3D12::D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT as usize];
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

        let raw_rasterizer = Direct3D12::D3D12_RASTERIZER_DESC {
            FillMode: conv::map_polygon_mode(desc.primitive.polygon_mode),
            CullMode: match desc.primitive.cull_mode {
                None => Direct3D12::D3D12_CULL_MODE_NONE,
                Some(wgt::Face::Front) => Direct3D12::D3D12_CULL_MODE_FRONT,
                Some(wgt::Face::Back) => Direct3D12::D3D12_CULL_MODE_BACK,
            },
            FrontCounterClockwise: match desc.primitive.front_face {
                wgt::FrontFace::Cw => Foundation::FALSE,
                wgt::FrontFace::Ccw => Foundation::TRUE,
            },
            DepthBias: bias.constant,
            DepthBiasClamp: bias.clamp,
            SlopeScaledDepthBias: bias.slope_scale,
            DepthClipEnable: Foundation::BOOL::from(!desc.primitive.unclipped_depth),
            MultisampleEnable: Foundation::BOOL::from(desc.multisample.count > 1),
            ForcedSampleCount: 0,
            AntialiasedLineEnable: false.into(),
            ConservativeRaster: if desc.primitive.conservative {
                Direct3D12::D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON
            } else {
                Direct3D12::D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
            },
        };

        let raw_desc = Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: unsafe {
                borrow_optional_interface_temporarily(&desc.layout.shared.signature)
            },
            VS: blob_vs.create_native_shader(),
            PS: match &blob_fs {
                Some(shader) => shader.create_native_shader(),
                None => Direct3D12::D3D12_SHADER_BYTECODE::default(),
            },
            GS: Direct3D12::D3D12_SHADER_BYTECODE::default(),
            DS: Direct3D12::D3D12_SHADER_BYTECODE::default(),
            HS: Direct3D12::D3D12_SHADER_BYTECODE::default(),
            StreamOutput: Direct3D12::D3D12_STREAM_OUTPUT_DESC {
                pSODeclaration: ptr::null(),
                NumEntries: 0,
                pBufferStrides: ptr::null(),
                NumStrides: 0,
                RasterizedStream: 0,
            },
            BlendState: Direct3D12::D3D12_BLEND_DESC {
                AlphaToCoverageEnable: Foundation::BOOL::from(
                    desc.multisample.alpha_to_coverage_enabled,
                ),
                IndependentBlendEnable: true.into(),
                RenderTarget: conv::map_render_targets(desc.color_targets),
            },
            SampleMask: desc.multisample.mask as u32,
            RasterizerState: raw_rasterizer,
            DepthStencilState: match desc.depth_stencil {
                Some(ref ds) => conv::map_depth_stencil(ds),
                None => Default::default(),
            },
            InputLayout: Direct3D12::D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: if input_element_descs.is_empty() {
                    ptr::null()
                } else {
                    input_element_descs.as_ptr()
                },
                NumElements: input_element_descs.len() as u32,
            },
            IBStripCutValue: match desc.primitive.strip_index_format {
                Some(wgt::IndexFormat::Uint16) => {
                    Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
                }
                Some(wgt::IndexFormat::Uint32) => {
                    Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF
                }
                None => Direct3D12::D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            },
            PrimitiveTopologyType: topology_class,
            NumRenderTargets: desc.color_targets.len() as u32,
            RTVFormats: rtv_formats,
            DSVFormat: desc
                .depth_stencil
                .as_ref()
                .map_or(Dxgi::Common::DXGI_FORMAT_UNKNOWN, |ds| {
                    auxil::dxgi::conv::map_texture_format(ds.format)
                }),
            SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: desc.multisample.count,
                Quality: 0,
            },
            NodeMask: 0,
            CachedPSO: Direct3D12::D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: ptr::null(),
                CachedBlobSizeInBytes: 0,
            },
            Flags: Direct3D12::D3D12_PIPELINE_STATE_FLAG_NONE,
        };

        let raw: Direct3D12::ID3D12PipelineState = {
            profiling::scope!("ID3D12Device::CreateGraphicsPipelineState");
            unsafe { self.raw.CreateGraphicsPipelineState(&raw_desc) }
        }
        .map_err(|err| crate::PipelineError::Linkage(shader_stages, err.to_string()))?;

        unsafe { blob_vs.destroy() };
        if let Some(blob_fs) = blob_fs {
            unsafe { blob_fs.destroy() };
        };

        if let Some(label) = desc.label {
            unsafe { raw.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.render_pipelines.add(1);

        Ok(super::RenderPipeline {
            raw,
            layout: desc.layout.shared.clone(),
            topology,
            vertex_strides,
        })
    }
    unsafe fn destroy_render_pipeline(&self, _pipeline: super::RenderPipeline) {
        self.counters.render_pipelines.sub(1);
    }

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<
            super::PipelineLayout,
            super::ShaderModule,
            super::PipelineCache,
        >,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        let blob_cs =
            self.load_shader(&desc.stage, desc.layout, naga::ShaderStage::Compute, None)?;

        let pair = {
            profiling::scope!("ID3D12Device::CreateComputePipelineState");
            unsafe {
                self.raw.CreateComputePipelineState(
                    &Direct3D12::D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: borrow_optional_interface_temporarily(
                            &desc.layout.shared.signature,
                        ),
                        CS: blob_cs.create_native_shader(),
                        NodeMask: 0,
                        CachedPSO: Direct3D12::D3D12_CACHED_PIPELINE_STATE::default(),
                        Flags: Direct3D12::D3D12_PIPELINE_STATE_FLAG_NONE,
                    },
                )
            }
        };

        unsafe { blob_cs.destroy() };

        let raw: Direct3D12::ID3D12PipelineState = pair.map_err(|err| {
            crate::PipelineError::Linkage(wgt::ShaderStages::COMPUTE, err.to_string())
        })?;

        if let Some(label) = desc.label {
            unsafe { raw.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.compute_pipelines.add(1);

        Ok(super::ComputePipeline {
            raw,
            layout: desc.layout.shared.clone(),
        })
    }

    unsafe fn destroy_compute_pipeline(&self, _pipeline: super::ComputePipeline) {
        self.counters.compute_pipelines.sub(1);
    }

    unsafe fn create_pipeline_cache(
        &self,
        _desc: &crate::PipelineCacheDescriptor<'_>,
    ) -> Result<super::PipelineCache, crate::PipelineCacheError> {
        Ok(super::PipelineCache)
    }
    unsafe fn destroy_pipeline_cache(&self, _: super::PipelineCache) {}

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<super::QuerySet, crate::DeviceError> {
        let (heap_ty, raw_ty) = match desc.ty {
            wgt::QueryType::Occlusion => (
                Direct3D12::D3D12_QUERY_HEAP_TYPE_OCCLUSION,
                Direct3D12::D3D12_QUERY_TYPE_BINARY_OCCLUSION,
            ),
            wgt::QueryType::PipelineStatistics(_) => (
                Direct3D12::D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
                Direct3D12::D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
            ),
            wgt::QueryType::Timestamp => (
                Direct3D12::D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Direct3D12::D3D12_QUERY_TYPE_TIMESTAMP,
            ),
        };

        let mut raw = None::<Direct3D12::ID3D12QueryHeap>;
        unsafe {
            self.raw.CreateQueryHeap(
                &Direct3D12::D3D12_QUERY_HEAP_DESC {
                    Type: heap_ty,
                    Count: desc.count,
                    NodeMask: 0,
                },
                &mut raw,
            )
        }
        .into_device_result("Query heap creation")?;

        let raw = raw.ok_or(crate::DeviceError::Unexpected)?;

        if let Some(label) = desc.label {
            unsafe { raw.SetName(&windows::core::HSTRING::from(label)) }
                .into_device_result("SetName")?;
        }

        self.counters.query_sets.add(1);

        Ok(super::QuerySet { raw, raw_ty })
    }

    unsafe fn destroy_query_set(&self, _set: super::QuerySet) {
        self.counters.query_sets.sub(1);
    }

    unsafe fn create_fence(&self) -> Result<super::Fence, crate::DeviceError> {
        let raw: Direct3D12::ID3D12Fence =
            unsafe { self.raw.CreateFence(0, Direct3D12::D3D12_FENCE_FLAG_SHARED) }
                .into_device_result("Fence creation")?;

        self.counters.fences.add(1);

        Ok(super::Fence { raw })
    }
    unsafe fn destroy_fence(&self, _fence: super::Fence) {
        self.counters.fences.sub(1);
    }

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
        let timeout_duration = Duration::from_millis(timeout_ms as u64);

        // We first check if the fence has already reached the value we're waiting for.
        let mut fence_value = unsafe { fence.raw.GetCompletedValue() };
        if fence_value >= value {
            return Ok(true);
        }

        unsafe { fence.raw.SetEventOnCompletion(value, self.idler.event.0) }
            .into_device_result("Set event")?;

        let start_time = Instant::now();

        // We need to loop to get correct behavior when timeouts are involved.
        //
        // wait(0):
        //   - We set the event from the fence value 0.
        //   - WaitForSingleObject times out, we return false.
        //
        // wait(1):
        //   - We set the event from the fence value 1.
        //   - WaitForSingleObject returns. However we do not know if the fence value is 0 or 1,
        //     just that _something_ triggered the event. We check the fence value, and if it is
        //     1, we return true. Otherwise, we loop and wait again.
        loop {
            let elapsed = start_time.elapsed();

            // We need to explicitly use checked_sub. Overflow with duration panics, and if the
            // timing works out just right, we can get a negative remaining wait duration.
            //
            // This happens when a previous iteration WaitForSingleObject succeeded with a previous fence value,
            // right before the timeout would have been hit.
            let remaining_wait_duration = match timeout_duration.checked_sub(elapsed) {
                Some(remaining) => remaining,
                None => {
                    log::trace!("Timeout elapsed in between waits!");
                    break Ok(false);
                }
            };

            log::trace!(
                "Waiting for fence value {} for {:?}",
                value,
                remaining_wait_duration
            );

            match unsafe {
                Threading::WaitForSingleObject(
                    self.idler.event.0,
                    remaining_wait_duration.as_millis().try_into().unwrap(),
                )
            } {
                Foundation::WAIT_OBJECT_0 => {}
                Foundation::WAIT_ABANDONED | Foundation::WAIT_FAILED => {
                    log::error!("Wait failed!");
                    break Err(crate::DeviceError::Lost);
                }
                Foundation::WAIT_TIMEOUT => {
                    log::trace!("Wait timed out!");
                    break Ok(false);
                }
                other => {
                    log::error!("Unexpected wait status: 0x{:?}", other);
                    break Err(crate::DeviceError::Lost);
                }
            };

            fence_value = unsafe { fence.raw.GetCompletedValue() };
            log::trace!("Wait complete! Fence actual value: {}", fence_value);

            if fence_value >= value {
                break Ok(true);
            }
        }
    }

    unsafe fn start_capture(&self) -> bool {
        #[cfg(feature = "renderdoc")]
        {
            unsafe {
                self.render_doc
                    .start_frame_capture(self.raw.as_raw(), ptr::null_mut())
            }
        }
        #[cfg(not(feature = "renderdoc"))]
        false
    }

    unsafe fn stop_capture(&self) {
        #[cfg(feature = "renderdoc")]
        unsafe {
            self.render_doc
                .end_frame_capture(self.raw.as_raw(), ptr::null_mut())
        }
    }

    unsafe fn get_acceleration_structure_build_sizes<'a>(
        &self,
        _desc: &crate::GetAccelerationStructureBuildSizesDescriptor<'a, super::Buffer>,
    ) -> crate::AccelerationStructureBuildSizes {
        // Implement using `GetRaytracingAccelerationStructurePrebuildInfo`:
        // https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#getraytracingaccelerationstructureprebuildinfo
        todo!()
    }

    unsafe fn get_acceleration_structure_device_address(
        &self,
        _acceleration_structure: &super::AccelerationStructure,
    ) -> wgt::BufferAddress {
        // Implement using `GetGPUVirtualAddress`:
        // https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12resource-getgpuvirtualaddress
        todo!()
    }

    unsafe fn create_acceleration_structure(
        &self,
        _desc: &crate::AccelerationStructureDescriptor,
    ) -> Result<super::AccelerationStructure, crate::DeviceError> {
        // Create a D3D12 resource as per-usual.
        todo!()
    }

    unsafe fn destroy_acceleration_structure(
        &self,
        _acceleration_structure: super::AccelerationStructure,
    ) {
        // Destroy a D3D12 resource as per-usual.
        todo!()
    }

    fn get_internal_counters(&self) -> wgt::HalCounters {
        self.counters.clone()
    }

    fn generate_allocator_report(&self) -> Option<wgt::AllocatorReport> {
        let mut upstream = self.mem_allocator.lock().allocator.generate_report();

        let allocations = upstream
            .allocations
            .iter_mut()
            .map(|alloc| wgt::AllocationReport {
                name: mem::take(&mut alloc.name),
                offset: alloc.offset,
                size: alloc.size,
            })
            .collect();

        let blocks = upstream
            .blocks
            .iter()
            .map(|block| wgt::MemoryBlockReport {
                size: block.size,
                allocations: block.allocations.clone(),
            })
            .collect();

        Some(wgt::AllocatorReport {
            allocations,
            blocks,
            total_allocated_bytes: upstream.total_allocated_bytes,
            total_reserved_bytes: upstream.total_reserved_bytes,
        })
    }
}
