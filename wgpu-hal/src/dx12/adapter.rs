use crate::{
    auxil::{self, dxgi::result::HResult as _},
    dx12::SurfaceTarget,
};
use std::{mem, ptr, sync::Arc, thread};
use winapi::{
    shared::{dxgi, dxgi1_2, minwindef::DWORD, windef, winerror},
    um::{d3d12 as d3d12_ty, d3d12sdklayers, winuser},
};

impl Drop for super::Adapter {
    fn drop(&mut self) {
        // Debug tracking alive objects
        if !thread::panicking()
            && self
                .private_caps
                .instance_flags
                .contains(crate::InstanceFlags::VALIDATION)
        {
            unsafe {
                self.report_live_objects();
            }
        }
        unsafe {
            self.raw.destroy();
        }
    }
}

impl super::Adapter {
    pub unsafe fn report_live_objects(&self) {
        if let Ok(debug_device) = unsafe {
            self.raw
                .cast::<d3d12sdklayers::ID3D12DebugDevice>()
                .into_result()
        } {
            unsafe {
                debug_device.ReportLiveDeviceObjects(
                    d3d12sdklayers::D3D12_RLDO_SUMMARY | d3d12sdklayers::D3D12_RLDO_IGNORE_INTERNAL,
                )
            };
            unsafe { debug_device.destroy() };
        }
    }

    pub fn raw_adapter(&self) -> &d3d12::DxgiAdapter {
        &self.raw
    }

    #[allow(trivial_casts)]
    pub(super) fn expose(
        adapter: d3d12::DxgiAdapter,
        library: &Arc<d3d12::D3D12Lib>,
        instance_flags: crate::InstanceFlags,
        dx12_shader_compiler: &wgt::Dx12Compiler,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        // Create the device so that we can get the capabilities.
        let device = {
            profiling::scope!("ID3D12Device::create_device");
            match library.create_device(*adapter, d3d12::FeatureLevel::L11_0) {
                Ok(pair) => match pair.into_result() {
                    Ok(device) => device,
                    Err(err) => {
                        log::warn!("Device creation failed: {}", err);
                        return None;
                    }
                },
                Err(err) => {
                    log::warn!("Device creation function is not found: {:?}", err);
                    return None;
                }
            }
        };

        profiling::scope!("feature queries");

        // We have found a possible adapter.
        // Acquire the device information.
        let mut desc: dxgi1_2::DXGI_ADAPTER_DESC2 = unsafe { mem::zeroed() };
        unsafe {
            adapter.unwrap_adapter2().GetDesc2(&mut desc);
        }

        let device_name = {
            use std::{ffi::OsString, os::windows::ffi::OsStringExt};
            let len = desc.Description.iter().take_while(|&&c| c != 0).count();
            let name = OsString::from_wide(&desc.Description[..len]);
            name.to_string_lossy().into_owned()
        };

        let mut features_architecture: d3d12_ty::D3D12_FEATURE_DATA_ARCHITECTURE =
            unsafe { mem::zeroed() };
        assert_eq!(0, unsafe {
            device.CheckFeatureSupport(
                d3d12_ty::D3D12_FEATURE_ARCHITECTURE,
                &mut features_architecture as *mut _ as *mut _,
                mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_ARCHITECTURE>() as _,
            )
        });

        let mut shader_model_support: d3d12_ty::D3D12_FEATURE_DATA_SHADER_MODEL =
            d3d12_ty::D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: d3d12_ty::D3D_SHADER_MODEL_6_0,
            };
        assert_eq!(0, unsafe {
            device.CheckFeatureSupport(
                d3d12_ty::D3D12_FEATURE_SHADER_MODEL,
                &mut shader_model_support as *mut _ as *mut _,
                mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_SHADER_MODEL>() as _,
            )
        });

        let mut workarounds = super::Workarounds::default();

        let info = wgt::AdapterInfo {
            backend: wgt::Backend::Dx12,
            name: device_name,
            vendor: desc.VendorId as usize,
            device: desc.DeviceId as usize,
            device_type: if (desc.Flags & dxgi::DXGI_ADAPTER_FLAG_SOFTWARE) != 0 {
                workarounds.avoid_cpu_descriptor_overwrites = true;
                wgt::DeviceType::Cpu
            } else if features_architecture.UMA != 0 {
                wgt::DeviceType::IntegratedGpu
            } else {
                wgt::DeviceType::DiscreteGpu
            },
            driver: String::new(),
            driver_info: String::new(),
        };

        let mut options: d3d12_ty::D3D12_FEATURE_DATA_D3D12_OPTIONS = unsafe { mem::zeroed() };
        assert_eq!(0, unsafe {
            device.CheckFeatureSupport(
                d3d12_ty::D3D12_FEATURE_D3D12_OPTIONS,
                &mut options as *mut _ as *mut _,
                mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_D3D12_OPTIONS>() as _,
            )
        });

        let _depth_bounds_test_supported = {
            let mut features2: d3d12_ty::D3D12_FEATURE_DATA_D3D12_OPTIONS2 =
                unsafe { mem::zeroed() };
            let hr = unsafe {
                device.CheckFeatureSupport(
                    d3d12_ty::D3D12_FEATURE_D3D12_OPTIONS2,
                    &mut features2 as *mut _ as *mut _,
                    mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_D3D12_OPTIONS2>() as _,
                )
            };
            hr == 0 && features2.DepthBoundsTestSupported != 0
        };

        let casting_fully_typed_format_supported = {
            let mut features3: crate::dx12::types::D3D12_FEATURE_DATA_D3D12_OPTIONS3 =
                unsafe { mem::zeroed() };
            let hr = unsafe {
                device.CheckFeatureSupport(
                    21, // D3D12_FEATURE_D3D12_OPTIONS3
                    &mut features3 as *mut _ as *mut _,
                    mem::size_of::<crate::dx12::types::D3D12_FEATURE_DATA_D3D12_OPTIONS3>() as _,
                )
            };
            hr == 0 && features3.CastingFullyTypedFormatSupported != 0
        };

        let private_caps = super::PrivateCapabilities {
            instance_flags,
            heterogeneous_resource_heaps: options.ResourceHeapTier
                != d3d12_ty::D3D12_RESOURCE_HEAP_TIER_1,
            memory_architecture: if features_architecture.UMA != 0 {
                super::MemoryArchitecture::Unified {
                    cache_coherent: features_architecture.CacheCoherentUMA != 0,
                }
            } else {
                super::MemoryArchitecture::NonUnified
            },
            heap_create_not_zeroed: false, //TODO: winapi support for Options7
            casting_fully_typed_format_supported,
        };

        // Theoretically vram limited, but in practice 2^20 is the limit
        let tier3_practical_descriptor_limit = 1 << 20;

        let (full_heap_count, _uav_count) = match options.ResourceBindingTier {
            d3d12_ty::D3D12_RESOURCE_BINDING_TIER_1 => (
                d3d12_ty::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1,
                8, // conservative, is 64 on feature level 11.1
            ),
            d3d12_ty::D3D12_RESOURCE_BINDING_TIER_2 => (
                d3d12_ty::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2,
                64,
            ),
            d3d12_ty::D3D12_RESOURCE_BINDING_TIER_3 => (
                tier3_practical_descriptor_limit,
                tier3_practical_descriptor_limit,
            ),
            other => {
                log::warn!("Unknown resource binding tier {}", other);
                (
                    d3d12_ty::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1,
                    8,
                )
            }
        };

        let mut features = wgt::Features::empty()
            | wgt::Features::DEPTH_CLIP_CONTROL
            | wgt::Features::DEPTH32FLOAT_STENCIL8
            | wgt::Features::INDIRECT_FIRST_INSTANCE
            | wgt::Features::MAPPABLE_PRIMARY_BUFFERS
            | wgt::Features::MULTI_DRAW_INDIRECT
            | wgt::Features::MULTI_DRAW_INDIRECT_COUNT
            | wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER
            | wgt::Features::ADDRESS_MODE_CLAMP_TO_ZERO
            | wgt::Features::POLYGON_MODE_LINE
            | wgt::Features::POLYGON_MODE_POINT
            | wgt::Features::VERTEX_WRITABLE_STORAGE
            | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgt::Features::TIMESTAMP_QUERY
            | wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgt::Features::TEXTURE_COMPRESSION_BC
            | wgt::Features::CLEAR_TEXTURE
            | wgt::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgt::Features::PUSH_CONSTANTS
            | wgt::Features::SHADER_PRIMITIVE_INDEX
            | wgt::Features::RG11B10UFLOAT_RENDERABLE;
        //TODO: in order to expose this, we need to run a compute shader
        // that extract the necessary statistics out of the D3D12 result.
        // Alternatively, we could allocate a buffer for the query set,
        // write the results there, and issue a bunch of copy commands.
        //| wgt::Features::PIPELINE_STATISTICS_QUERY

        features.set(
            wgt::Features::CONSERVATIVE_RASTERIZATION,
            options.ConservativeRasterizationTier
                != d3d12_ty::D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED,
        );

        features.set(
            wgt::Features::TEXTURE_BINDING_ARRAY
                | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            shader_model_support.HighestShaderModel >= d3d12_ty::D3D_SHADER_MODEL_5_1,
        );

        // TODO: Determine if IPresentationManager is supported
        let presentation_timer = auxil::dxgi::time::PresentationTimer::new_dxgi();

        let base = wgt::Limits::default();

        Some(crate::ExposedAdapter {
            adapter: super::Adapter {
                raw: adapter,
                device,
                library: Arc::clone(library),
                private_caps,
                presentation_timer,
                workarounds,
                dx12_shader_compiler: dx12_shader_compiler.clone(),
            },
            info,
            features,
            capabilities: crate::Capabilities {
                limits: wgt::Limits {
                    max_texture_dimension_1d: d3d12_ty::D3D12_REQ_TEXTURE1D_U_DIMENSION,
                    max_texture_dimension_2d: d3d12_ty::D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
                        .min(d3d12_ty::D3D12_REQ_TEXTURECUBE_DIMENSION),
                    max_texture_dimension_3d: d3d12_ty::D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                    max_texture_array_layers: d3d12_ty::D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
                    max_bind_groups: crate::MAX_BIND_GROUPS as u32,
                    max_bindings_per_bind_group: 65535,
                    // dynamic offsets take a root constant, so we expose the minimum here
                    max_dynamic_uniform_buffers_per_pipeline_layout: base
                        .max_dynamic_uniform_buffers_per_pipeline_layout,
                    max_dynamic_storage_buffers_per_pipeline_layout: base
                        .max_dynamic_storage_buffers_per_pipeline_layout,
                    max_sampled_textures_per_shader_stage: match options.ResourceBindingTier {
                        d3d12_ty::D3D12_RESOURCE_BINDING_TIER_1 => 128,
                        _ => full_heap_count,
                    },
                    max_samplers_per_shader_stage: match options.ResourceBindingTier {
                        d3d12_ty::D3D12_RESOURCE_BINDING_TIER_1 => 16,
                        _ => d3d12_ty::D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
                    },
                    // these both account towards `uav_count`, but we can't express the limit as as sum
                    max_storage_buffers_per_shader_stage: base.max_storage_buffers_per_shader_stage,
                    max_storage_textures_per_shader_stage: base
                        .max_storage_textures_per_shader_stage,
                    max_uniform_buffers_per_shader_stage: full_heap_count,
                    max_uniform_buffer_binding_size:
                        d3d12_ty::D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16,
                    max_storage_buffer_binding_size: crate::auxil::MAX_I32_BINDING_SIZE,
                    max_vertex_buffers: d3d12_ty::D3D12_VS_INPUT_REGISTER_COUNT
                        .min(crate::MAX_VERTEX_BUFFERS as u32),
                    max_vertex_attributes: d3d12_ty::D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT,
                    max_vertex_buffer_array_stride: d3d12_ty::D3D12_SO_BUFFER_MAX_STRIDE_IN_BYTES,
                    // The push constants are part of the root signature which
                    // has a limit of 64 DWORDS (256 bytes), but other resources
                    // also share the root signature:
                    //
                    // - push constants consume a `DWORD` for each `4 bytes` of data
                    // - If a bind group has buffers it will consume a `DWORD`
                    //   for the descriptor table
                    // - If a bind group has samplers it will consume a `DWORD`
                    //   for the descriptor table
                    // - Each dynamic buffer will consume `2 DWORDs` for the
                    //   root descriptor
                    // - The special constants buffer count as constants
                    //
                    // Since we can't know beforehand all root signatures that
                    // will be created, the max size to be used for push
                    // constants needs to be set to a reasonable number instead.
                    //
                    // Source: https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits#memory-limits-and-costs
                    max_push_constant_size: 128,
                    min_uniform_buffer_offset_alignment:
                        d3d12_ty::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
                    min_storage_buffer_offset_alignment: 4,
                    max_inter_stage_shader_components: base.max_inter_stage_shader_components,
                    max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size, //TODO?
                    max_compute_invocations_per_workgroup:
                        d3d12_ty::D3D12_CS_4_X_THREAD_GROUP_MAX_THREADS_PER_GROUP,
                    max_compute_workgroup_size_x: d3d12_ty::D3D12_CS_THREAD_GROUP_MAX_X,
                    max_compute_workgroup_size_y: d3d12_ty::D3D12_CS_THREAD_GROUP_MAX_Y,
                    max_compute_workgroup_size_z: d3d12_ty::D3D12_CS_THREAD_GROUP_MAX_Z,
                    max_compute_workgroups_per_dimension:
                        d3d12_ty::D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                    max_buffer_size: u64::MAX,
                },
                alignments: crate::Alignments {
                    buffer_copy_offset: wgt::BufferSize::new(
                        d3d12_ty::D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64,
                    )
                    .unwrap(),
                    buffer_copy_pitch: wgt::BufferSize::new(
                        d3d12_ty::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64,
                    )
                    .unwrap(),
                },
                downlevel: wgt::DownlevelCapabilities::default(),
            },
        })
    }
}

impl crate::Adapter<super::Api> for super::Adapter {
    unsafe fn open(
        &self,
        _features: wgt::Features,
        _limits: &wgt::Limits,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        let queue = {
            profiling::scope!("ID3D12Device::CreateCommandQueue");
            self.device
                .create_command_queue(
                    d3d12::CmdListType::Direct,
                    d3d12::Priority::Normal,
                    d3d12::CommandQueueFlags::empty(),
                    0,
                )
                .into_device_result("Queue creation")?
        };

        let device = super::Device::new(
            self.device,
            queue,
            self.private_caps,
            &self.library,
            self.dx12_shader_compiler.clone(),
        )?;
        Ok(crate::OpenDevice {
            device,
            queue: super::Queue {
                raw: queue,
                temp_lists: Vec::new(),
            },
        })
    }

    #[allow(trivial_casts)]
    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        use crate::TextureFormatCapabilities as Tfc;

        let raw_format = match auxil::dxgi::conv::map_texture_format_failable(format) {
            Some(f) => f,
            None => return Tfc::empty(),
        };
        let srv_uav_format = if format.is_combined_depth_stencil_format() {
            auxil::dxgi::conv::map_texture_format_for_srv_uav(
                format,
                // use the depth aspect here as opposed to stencil since it has more capabilities
                crate::FormatAspects::DEPTH,
            )
        } else {
            auxil::dxgi::conv::map_texture_format_for_srv_uav(
                format,
                crate::FormatAspects::from(format),
            )
        }
        .unwrap();

        let mut data = d3d12_ty::D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: raw_format,
            Support1: unsafe { mem::zeroed() },
            Support2: unsafe { mem::zeroed() },
        };
        assert_eq!(winerror::S_OK, unsafe {
            self.device.CheckFeatureSupport(
                d3d12_ty::D3D12_FEATURE_FORMAT_SUPPORT,
                &mut data as *mut _ as *mut _,
                mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as _,
            )
        });

        // Because we use a different format for SRV and UAV views of depth textures, we need to check
        // the features that use SRV/UAVs using the no-depth format.
        let mut data_srv_uav = d3d12_ty::D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: srv_uav_format,
            Support1: d3d12_ty::D3D12_FORMAT_SUPPORT1_NONE,
            Support2: d3d12_ty::D3D12_FORMAT_SUPPORT2_NONE,
        };
        if raw_format != srv_uav_format {
            // Only-recheck if we're using a different format
            assert_eq!(winerror::S_OK, unsafe {
                self.device.CheckFeatureSupport(
                    d3d12_ty::D3D12_FEATURE_FORMAT_SUPPORT,
                    ptr::addr_of_mut!(data_srv_uav).cast(),
                    DWORD::try_from(mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_FORMAT_SUPPORT>())
                        .unwrap(),
                )
            });
        } else {
            // Same format, just copy over.
            data_srv_uav = data;
        }

        let mut caps = Tfc::COPY_SRC | Tfc::COPY_DST;
        let is_texture = data.Support1
            & (d3d12_ty::D3D12_FORMAT_SUPPORT1_TEXTURE1D
                | d3d12_ty::D3D12_FORMAT_SUPPORT1_TEXTURE2D
                | d3d12_ty::D3D12_FORMAT_SUPPORT1_TEXTURE3D
                | d3d12_ty::D3D12_FORMAT_SUPPORT1_TEXTURECUBE)
            != 0;
        // SRVs use srv_uav_format
        caps.set(
            Tfc::SAMPLED,
            is_texture && data_srv_uav.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_SHADER_LOAD != 0,
        );
        caps.set(
            Tfc::SAMPLED_LINEAR,
            data_srv_uav.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE != 0,
        );
        caps.set(
            Tfc::COLOR_ATTACHMENT,
            data.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_RENDER_TARGET != 0,
        );
        caps.set(
            Tfc::COLOR_ATTACHMENT_BLEND,
            data.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_BLENDABLE != 0,
        );
        caps.set(
            Tfc::DEPTH_STENCIL_ATTACHMENT,
            data.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL != 0,
        );
        // UAVs use srv_uav_format
        caps.set(
            Tfc::STORAGE,
            data_srv_uav.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW
                != 0,
        );
        caps.set(
            Tfc::STORAGE_READ_WRITE,
            data_srv_uav.Support2 & d3d12_ty::D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD != 0,
        );

        // We load via UAV/SRV so use srv_uav_format
        let no_msaa_load = caps.contains(Tfc::SAMPLED)
            && data_srv_uav.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD == 0;

        let no_msaa_target = data.Support1
            & (d3d12_ty::D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                | d3d12_ty::D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)
            != 0
            && data.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET == 0;

        caps.set(
            Tfc::MULTISAMPLE_RESOLVE,
            data.Support1 & d3d12_ty::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE != 0,
        );

        let mut ms_levels = d3d12_ty::D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: raw_format,
            SampleCount: 0,
            Flags: d3d12_ty::D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
            NumQualityLevels: 0,
        };

        let mut set_sample_count = |sc: u32, tfc: Tfc| {
            ms_levels.SampleCount = sc;

            if unsafe {
                self.device.CheckFeatureSupport(
                    d3d12_ty::D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                    <*mut _>::cast(&mut ms_levels),
                    mem::size_of::<d3d12_ty::D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as _,
                )
            } == winerror::S_OK
                && ms_levels.NumQualityLevels != 0
            {
                caps.set(tfc, !no_msaa_load && !no_msaa_target);
            }
        };

        set_sample_count(2, Tfc::MULTISAMPLE_X2);
        set_sample_count(4, Tfc::MULTISAMPLE_X4);
        set_sample_count(8, Tfc::MULTISAMPLE_X8);
        set_sample_count(16, Tfc::MULTISAMPLE_X16);

        caps
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &super::Surface,
    ) -> Option<crate::SurfaceCapabilities> {
        let current_extent = {
            match surface.target {
                SurfaceTarget::WndHandle(wnd_handle) => {
                    let mut rect: windef::RECT = unsafe { mem::zeroed() };
                    if unsafe { winuser::GetClientRect(wnd_handle, &mut rect) } != 0 {
                        Some(wgt::Extent3d {
                            width: (rect.right - rect.left) as u32,
                            height: (rect.bottom - rect.top) as u32,
                            depth_or_array_layers: 1,
                        })
                    } else {
                        log::warn!("Unable to get the window client rect");
                        None
                    }
                }
                SurfaceTarget::Visual(_) | SurfaceTarget::SurfaceHandle(_) => None,
            }
        };

        let mut present_modes = vec![wgt::PresentMode::Mailbox, wgt::PresentMode::Fifo];
        if surface.supports_allow_tearing {
            present_modes.push(wgt::PresentMode::Immediate);
        }

        Some(crate::SurfaceCapabilities {
            formats: vec![
                wgt::TextureFormat::Bgra8UnormSrgb,
                wgt::TextureFormat::Bgra8Unorm,
                wgt::TextureFormat::Rgba8UnormSrgb,
                wgt::TextureFormat::Rgba8Unorm,
                wgt::TextureFormat::Rgb10a2Unorm,
                wgt::TextureFormat::Rgba16Float,
            ],
            // we currently use a flip effect which supports 2..=16 buffers
            swap_chain_sizes: 2..=16,
            current_extent,
            // TODO: figure out the exact bounds
            extents: wgt::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            }..=wgt::Extent3d {
                width: 4096,
                height: 4096,
                depth_or_array_layers: 1,
            },
            usage: crate::TextureUses::COLOR_TARGET
                | crate::TextureUses::COPY_SRC
                | crate::TextureUses::COPY_DST,
            present_modes,
            composite_alpha_modes: vec![wgt::CompositeAlphaMode::Opaque],
        })
    }

    unsafe fn get_presentation_timestamp(&self) -> wgt::PresentationTimestamp {
        wgt::PresentationTimestamp(self.presentation_timer.get_timestamp_ns())
    }
}
