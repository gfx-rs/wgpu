use crate::{
    auxil::{self, dxgi::result::HResult as _},
    dx12::SurfaceTarget,
};
use std::{mem, sync::Arc, thread};
use winapi::{
    shared::{dxgi, dxgi1_2, dxgi1_5, minwindef, windef, winerror},
    um::{d3d12, d3d12sdklayers, winuser},
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
        if let Ok(debug_device) = self
            .raw
            .cast::<d3d12sdklayers::ID3D12DebugDevice>()
            .into_result()
        {
            debug_device.ReportLiveDeviceObjects(
                d3d12sdklayers::D3D12_RLDO_SUMMARY | d3d12sdklayers::D3D12_RLDO_IGNORE_INTERNAL,
            );
            debug_device.destroy();
        }
    }

    pub fn raw_adapter(&self) -> &native::DxgiAdapter {
        &self.raw
    }

    #[allow(trivial_casts)]
    pub(super) fn expose(
        adapter: native::DxgiAdapter,
        library: &Arc<native::D3D12Lib>,
        instance_flags: crate::InstanceFlags,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        // Create the device so that we can get the capabilities.
        let device = {
            profiling::scope!("ID3D12Device::create_device");
            match library.create_device(*adapter, native::FeatureLevel::L11_0) {
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

        let mut features_architecture: d3d12::D3D12_FEATURE_DATA_ARCHITECTURE =
            unsafe { mem::zeroed() };
        assert_eq!(0, unsafe {
            device.CheckFeatureSupport(
                d3d12::D3D12_FEATURE_ARCHITECTURE,
                &mut features_architecture as *mut _ as *mut _,
                mem::size_of::<d3d12::D3D12_FEATURE_DATA_ARCHITECTURE>() as _,
            )
        });

        let mut shader_model_support: d3d12::D3D12_FEATURE_DATA_SHADER_MODEL =
            d3d12::D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: d3d12::D3D_SHADER_MODEL_6_0,
            };
        assert_eq!(0, unsafe {
            device.CheckFeatureSupport(
                d3d12::D3D12_FEATURE_SHADER_MODEL,
                &mut shader_model_support as *mut _ as *mut _,
                mem::size_of::<d3d12::D3D12_FEATURE_DATA_SHADER_MODEL>() as _,
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
        };

        let mut options: d3d12::D3D12_FEATURE_DATA_D3D12_OPTIONS = unsafe { mem::zeroed() };
        assert_eq!(0, unsafe {
            device.CheckFeatureSupport(
                d3d12::D3D12_FEATURE_D3D12_OPTIONS,
                &mut options as *mut _ as *mut _,
                mem::size_of::<d3d12::D3D12_FEATURE_DATA_D3D12_OPTIONS>() as _,
            )
        });

        let _depth_bounds_test_supported = {
            let mut features2: d3d12::D3D12_FEATURE_DATA_D3D12_OPTIONS2 = unsafe { mem::zeroed() };
            let hr = unsafe {
                device.CheckFeatureSupport(
                    d3d12::D3D12_FEATURE_D3D12_OPTIONS2,
                    &mut features2 as *mut _ as *mut _,
                    mem::size_of::<d3d12::D3D12_FEATURE_DATA_D3D12_OPTIONS2>() as _,
                )
            };
            hr == 0 && features2.DepthBoundsTestSupported != 0
        };

        //Note: `D3D12_FEATURE_D3D12_OPTIONS3::CastingFullyTypedFormatSupported` can be checked
        // to know if we can skip "typeless" formats entirely.

        let private_caps = super::PrivateCapabilities {
            instance_flags,
            heterogeneous_resource_heaps: options.ResourceHeapTier
                != d3d12::D3D12_RESOURCE_HEAP_TIER_1,
            memory_architecture: if features_architecture.UMA != 0 {
                super::MemoryArchitecture::Unified {
                    cache_coherent: features_architecture.CacheCoherentUMA != 0,
                }
            } else {
                super::MemoryArchitecture::NonUnified
            },
            heap_create_not_zeroed: false, //TODO: winapi support for Options7
        };

        // Theoretically vram limited, but in practice 2^20 is the limit
        let tier3_practical_descriptor_limit = 1 << 20;

        let (full_heap_count, _uav_count) = match options.ResourceBindingTier {
            d3d12::D3D12_RESOURCE_BINDING_TIER_1 => (
                d3d12::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1,
                8, // conservative, is 64 on feature level 11.1
            ),
            d3d12::D3D12_RESOURCE_BINDING_TIER_2 => (
                d3d12::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2,
                64,
            ),
            d3d12::D3D12_RESOURCE_BINDING_TIER_3 => (
                tier3_practical_descriptor_limit,
                tier3_practical_descriptor_limit,
            ),
            other => {
                log::warn!("Unknown resource binding tier {}", other);
                (
                    d3d12::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1,
                    8,
                )
            }
        };

        let mut features = wgt::Features::empty()
            | wgt::Features::DEPTH_CLIP_CONTROL
            | wgt::Features::DEPTH24UNORM_STENCIL8
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
            | wgt::Features::WRITE_TIMESTAMP_INSIDE_PASSES
            | wgt::Features::TEXTURE_COMPRESSION_BC
            | wgt::Features::CLEAR_TEXTURE
            | wgt::Features::TEXTURE_FORMAT_16BIT_NORM;
        //TODO: in order to expose this, we need to run a compute shader
        // that extract the necessary statistics out of the D3D12 result.
        // Alternatively, we could allocate a buffer for the query set,
        // write the results there, and issue a bunch of copy commands.
        //| wgt::Features::PIPELINE_STATISTICS_QUERY

        features.set(
            wgt::Features::CONSERVATIVE_RASTERIZATION,
            options.ConservativeRasterizationTier
                != d3d12::D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED,
        );

        features.set(
            wgt::Features::TEXTURE_BINDING_ARRAY
                | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            shader_model_support.HighestShaderModel >= d3d12::D3D_SHADER_MODEL_5_1,
        );

        let base = wgt::Limits::default();

        Some(crate::ExposedAdapter {
            adapter: super::Adapter {
                raw: adapter,
                device,
                library: Arc::clone(library),
                private_caps,
                workarounds,
            },
            info,
            features,
            capabilities: crate::Capabilities {
                limits: wgt::Limits {
                    max_texture_dimension_1d: d3d12::D3D12_REQ_TEXTURE1D_U_DIMENSION,
                    max_texture_dimension_2d: d3d12::D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
                        .min(d3d12::D3D12_REQ_TEXTURECUBE_DIMENSION),
                    max_texture_dimension_3d: d3d12::D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                    max_texture_array_layers: d3d12::D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
                    max_bind_groups: crate::MAX_BIND_GROUPS as u32,
                    // dynamic offsets take a root constant, so we expose the minimum here
                    max_dynamic_uniform_buffers_per_pipeline_layout: base
                        .max_dynamic_uniform_buffers_per_pipeline_layout,
                    max_dynamic_storage_buffers_per_pipeline_layout: base
                        .max_dynamic_storage_buffers_per_pipeline_layout,
                    max_sampled_textures_per_shader_stage: match options.ResourceBindingTier {
                        d3d12::D3D12_RESOURCE_BINDING_TIER_1 => 128,
                        _ => full_heap_count,
                    },
                    max_samplers_per_shader_stage: match options.ResourceBindingTier {
                        d3d12::D3D12_RESOURCE_BINDING_TIER_1 => 16,
                        _ => d3d12::D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
                    },
                    // these both account towards `uav_count`, but we can't express the limit as as sum
                    max_storage_buffers_per_shader_stage: base.max_storage_buffers_per_shader_stage,
                    max_storage_textures_per_shader_stage: base
                        .max_storage_textures_per_shader_stage,
                    max_uniform_buffers_per_shader_stage: full_heap_count,
                    max_uniform_buffer_binding_size: d3d12::D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT
                        * 16,
                    max_storage_buffer_binding_size: crate::auxil::MAX_I32_BINDING_SIZE,
                    max_vertex_buffers: d3d12::D3D12_VS_INPUT_REGISTER_COUNT
                        .min(crate::MAX_VERTEX_BUFFERS as u32),
                    max_vertex_attributes: d3d12::D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT,
                    max_vertex_buffer_array_stride: d3d12::D3D12_SO_BUFFER_MAX_STRIDE_IN_BYTES,
                    max_push_constant_size: 0,
                    min_uniform_buffer_offset_alignment:
                        d3d12::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
                    min_storage_buffer_offset_alignment: 4,
                    max_inter_stage_shader_components: base.max_inter_stage_shader_components,
                    max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size, //TODO?
                    max_compute_invocations_per_workgroup:
                        d3d12::D3D12_CS_4_X_THREAD_GROUP_MAX_THREADS_PER_GROUP,
                    max_compute_workgroup_size_x: d3d12::D3D12_CS_THREAD_GROUP_MAX_X,
                    max_compute_workgroup_size_y: d3d12::D3D12_CS_THREAD_GROUP_MAX_Y,
                    max_compute_workgroup_size_z: d3d12::D3D12_CS_THREAD_GROUP_MAX_Z,
                    max_compute_workgroups_per_dimension:
                        d3d12::D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                    max_buffer_size: u64::MAX,
                },
                alignments: crate::Alignments {
                    buffer_copy_offset: wgt::BufferSize::new(
                        d3d12::D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64,
                    )
                    .unwrap(),
                    buffer_copy_pitch: wgt::BufferSize::new(
                        d3d12::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64,
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
                    native::CmdListType::Direct,
                    native::Priority::Normal,
                    native::CommandQueueFlags::empty(),
                    0,
                )
                .into_device_result("Queue creation")?
        };

        let device = super::Device::new(self.device, queue, self.private_caps, &self.library)?;
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

        let raw_format = auxil::dxgi::conv::map_texture_format(format);
        let mut data = d3d12::D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: raw_format,
            Support1: mem::zeroed(),
            Support2: mem::zeroed(),
        };
        assert_eq!(
            winerror::S_OK,
            self.device.CheckFeatureSupport(
                d3d12::D3D12_FEATURE_FORMAT_SUPPORT,
                &mut data as *mut _ as *mut _,
                mem::size_of::<d3d12::D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as _,
            )
        );

        let mut caps = Tfc::COPY_SRC | Tfc::COPY_DST;
        let is_texture = data.Support1
            & (d3d12::D3D12_FORMAT_SUPPORT1_TEXTURE1D
                | d3d12::D3D12_FORMAT_SUPPORT1_TEXTURE2D
                | d3d12::D3D12_FORMAT_SUPPORT1_TEXTURE3D
                | d3d12::D3D12_FORMAT_SUPPORT1_TEXTURECUBE)
            != 0;
        caps.set(
            Tfc::SAMPLED,
            is_texture && data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_SHADER_LOAD != 0,
        );
        caps.set(
            Tfc::SAMPLED_LINEAR,
            data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE != 0,
        );
        caps.set(
            Tfc::COLOR_ATTACHMENT,
            data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_RENDER_TARGET != 0,
        );
        caps.set(
            Tfc::COLOR_ATTACHMENT_BLEND,
            data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_BLENDABLE != 0,
        );
        caps.set(
            Tfc::DEPTH_STENCIL_ATTACHMENT,
            data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL != 0,
        );
        caps.set(
            Tfc::STORAGE,
            data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW != 0,
        );
        caps.set(
            Tfc::STORAGE_READ_WRITE,
            data.Support2 & d3d12::D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD != 0,
        );

        let no_msaa_load = caps.contains(Tfc::SAMPLED)
            && data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD == 0;
        let no_msaa_target = data.Support1
            & (d3d12::D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                | d3d12::D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)
            != 0
            && data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET == 0;
        caps.set(Tfc::MULTISAMPLE, !no_msaa_load && !no_msaa_target);
        caps.set(
            Tfc::MULTISAMPLE_RESOLVE,
            data.Support1 & d3d12::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE != 0,
        );

        caps
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &super::Surface,
    ) -> Option<crate::SurfaceCapabilities> {
        let current_extent = {
            match surface.target {
                SurfaceTarget::WndHandle(wnd_handle) => {
                    let mut rect: windef::RECT = mem::zeroed();
                    if winuser::GetClientRect(wnd_handle, &mut rect) != 0 {
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
                SurfaceTarget::Visual(_) => None,
            }
        };

        let mut present_modes = vec![wgt::PresentMode::Fifo];
        #[allow(trivial_casts)]
        if let Some(factory5) = surface.factory.as_factory5() {
            let mut allow_tearing: minwindef::BOOL = minwindef::FALSE;
            let hr = factory5.CheckFeatureSupport(
                dxgi1_5::DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                &mut allow_tearing as *mut _ as *mut _,
                mem::size_of::<minwindef::BOOL>() as _,
            );

            match hr.into_result() {
                Err(err) => log::warn!("Unable to check for tearing support: {}", err),
                Ok(()) => present_modes.push(wgt::PresentMode::Immediate),
            }
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
            composite_alpha_modes: vec![
                crate::CompositeAlphaMode::Opaque,
                crate::CompositeAlphaMode::PreMultiplied,
                crate::CompositeAlphaMode::PostMultiplied,
            ],
        })
    }
}
