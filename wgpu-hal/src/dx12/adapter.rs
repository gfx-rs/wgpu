use std::{
    mem::{size_of, size_of_val},
    ptr,
    sync::Arc,
    thread,
};

use parking_lot::Mutex;
use windows::{
    core::Interface as _,
    Win32::{
        Graphics::{Direct3D, Direct3D12, Dxgi},
        UI::WindowsAndMessaging,
    },
};

use super::D3D12Lib;
use crate::{
    auxil::{
        self,
        dxgi::{factory::DxgiAdapter, result::HResult},
    },
    dx12::{shader_compilation, SurfaceTarget},
};

impl Drop for super::Adapter {
    fn drop(&mut self) {
        // Debug tracking alive objects
        if !thread::panicking()
            && self
                .private_caps
                .instance_flags
                .contains(wgt::InstanceFlags::VALIDATION)
        {
            unsafe {
                self.report_live_objects();
            }
        }
    }
}

impl super::Adapter {
    pub unsafe fn report_live_objects(&self) {
        if let Ok(debug_device) = self.raw.cast::<Direct3D12::ID3D12DebugDevice>() {
            unsafe {
                debug_device.ReportLiveDeviceObjects(
                    Direct3D12::D3D12_RLDO_SUMMARY | Direct3D12::D3D12_RLDO_IGNORE_INTERNAL,
                )
            }
            .unwrap()
        }
    }

    pub fn raw_adapter(&self) -> &DxgiAdapter {
        &self.raw
    }

    pub(super) fn expose(
        adapter: DxgiAdapter,
        library: &Arc<D3D12Lib>,
        instance_flags: wgt::InstanceFlags,
        dxc_container: Option<Arc<shader_compilation::DxcContainer>>,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        // Create the device so that we can get the capabilities.
        let device = {
            profiling::scope!("ID3D12Device::create_device");
            library
                .create_device(&adapter, Direct3D::D3D_FEATURE_LEVEL_11_0)
                .ok()??
        };

        profiling::scope!("feature queries");

        // Detect the highest supported feature level.
        let d3d_feature_level = [
            Direct3D::D3D_FEATURE_LEVEL_12_1,
            Direct3D::D3D_FEATURE_LEVEL_12_0,
            Direct3D::D3D_FEATURE_LEVEL_11_1,
            Direct3D::D3D_FEATURE_LEVEL_11_0,
        ];
        let mut device_levels = Direct3D12::D3D12_FEATURE_DATA_FEATURE_LEVELS {
            NumFeatureLevels: d3d_feature_level.len() as u32,
            pFeatureLevelsRequested: d3d_feature_level.as_ptr().cast(),
            MaxSupportedFeatureLevel: Default::default(),
        };
        unsafe {
            device.CheckFeatureSupport(
                Direct3D12::D3D12_FEATURE_FEATURE_LEVELS,
                <*mut _>::cast(&mut device_levels),
                size_of_val(&device_levels) as u32,
            )
        }
        .unwrap();
        let max_feature_level = device_levels.MaxSupportedFeatureLevel;

        // We have found a possible adapter.
        // Acquire the device information.
        let desc = unsafe { adapter.GetDesc2() }.unwrap();

        let device_name = auxil::dxgi::conv::map_adapter_name(desc.Description);

        let mut features_architecture = Direct3D12::D3D12_FEATURE_DATA_ARCHITECTURE::default();

        unsafe {
            device.CheckFeatureSupport(
                Direct3D12::D3D12_FEATURE_ARCHITECTURE,
                <*mut _>::cast(&mut features_architecture),
                size_of_val(&features_architecture) as u32,
            )
        }
        .unwrap();

        let mut workarounds = super::Workarounds::default();

        let info = wgt::AdapterInfo {
            backend: wgt::Backend::Dx12,
            name: device_name,
            vendor: desc.VendorId,
            device: desc.DeviceId,
            device_type: if Dxgi::DXGI_ADAPTER_FLAG(desc.Flags as i32)
                .contains(Dxgi::DXGI_ADAPTER_FLAG_SOFTWARE)
            {
                workarounds.avoid_cpu_descriptor_overwrites = true;
                wgt::DeviceType::Cpu
            } else if features_architecture.UMA.as_bool() {
                wgt::DeviceType::IntegratedGpu
            } else {
                wgt::DeviceType::DiscreteGpu
            },
            driver: {
                if let Ok(i) = unsafe { adapter.CheckInterfaceSupport(&Dxgi::IDXGIDevice::IID) } {
                    const MASK: i64 = 0xFFFF;
                    format!(
                        "{}.{}.{}.{}",
                        i >> 48,
                        (i >> 32) & MASK,
                        (i >> 16) & MASK,
                        i & MASK
                    )
                } else {
                    String::new()
                }
            },
            driver_info: String::new(),
        };

        let mut options = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        unsafe {
            device.CheckFeatureSupport(
                Direct3D12::D3D12_FEATURE_D3D12_OPTIONS,
                <*mut _>::cast(&mut options),
                size_of_val(&options) as u32,
            )
        }
        .unwrap();

        let _depth_bounds_test_supported = {
            let mut features2 = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            unsafe {
                device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_D3D12_OPTIONS2,
                    <*mut _>::cast(&mut features2),
                    size_of_val(&features2) as u32,
                )
            }
            .is_ok()
                && features2.DepthBoundsTestSupported.as_bool()
        };

        let casting_fully_typed_format_supported = {
            let mut features3 = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            unsafe {
                device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_D3D12_OPTIONS3,
                    <*mut _>::cast(&mut features3),
                    size_of_val(&features3) as u32,
                )
            }
            .is_ok()
                && features3.CastingFullyTypedFormatSupported.as_bool()
        };

        let heap_create_not_zeroed = {
            // For D3D12_HEAP_FLAG_CREATE_NOT_ZEROED we just need to
            // make sure that options7 can be queried. See also:
            // https://devblogs.microsoft.com/directx/coming-to-directx-12-more-control-over-memory-allocation/
            let mut features7 = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            unsafe {
                device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_D3D12_OPTIONS7,
                    <*mut _>::cast(&mut features7),
                    size_of_val(&features7) as u32,
                )
            }
            .is_ok()
        };

        let shader_model = if dxc_container.is_none() {
            naga::back::hlsl::ShaderModel::V5_1
        } else {
            let mut versions = [
                Direct3D12::D3D_SHADER_MODEL_6_7,
                Direct3D12::D3D_SHADER_MODEL_6_6,
                Direct3D12::D3D_SHADER_MODEL_6_5,
                Direct3D12::D3D_SHADER_MODEL_6_4,
                Direct3D12::D3D_SHADER_MODEL_6_3,
                Direct3D12::D3D_SHADER_MODEL_6_2,
                Direct3D12::D3D_SHADER_MODEL_6_1,
                Direct3D12::D3D_SHADER_MODEL_6_0,
                Direct3D12::D3D_SHADER_MODEL_5_1,
            ]
            .iter();
            match loop {
                if let Some(&sm) = versions.next() {
                    let mut sm = Direct3D12::D3D12_FEATURE_DATA_SHADER_MODEL {
                        HighestShaderModel: sm,
                    };
                    if unsafe {
                        device.CheckFeatureSupport(
                            Direct3D12::D3D12_FEATURE_SHADER_MODEL,
                            <*mut _>::cast(&mut sm),
                            size_of_val(&sm) as u32,
                        )
                    }
                    .is_ok()
                    {
                        break sm.HighestShaderModel;
                    }
                } else {
                    break Direct3D12::D3D_SHADER_MODEL_5_1;
                }
            } {
                Direct3D12::D3D_SHADER_MODEL_5_1 => naga::back::hlsl::ShaderModel::V5_1,
                Direct3D12::D3D_SHADER_MODEL_6_0 => naga::back::hlsl::ShaderModel::V6_0,
                Direct3D12::D3D_SHADER_MODEL_6_1 => naga::back::hlsl::ShaderModel::V6_1,
                Direct3D12::D3D_SHADER_MODEL_6_2 => naga::back::hlsl::ShaderModel::V6_2,
                Direct3D12::D3D_SHADER_MODEL_6_3 => naga::back::hlsl::ShaderModel::V6_3,
                Direct3D12::D3D_SHADER_MODEL_6_4 => naga::back::hlsl::ShaderModel::V6_4,
                Direct3D12::D3D_SHADER_MODEL_6_5 => naga::back::hlsl::ShaderModel::V6_5,
                Direct3D12::D3D_SHADER_MODEL_6_6 => naga::back::hlsl::ShaderModel::V6_6,
                Direct3D12::D3D_SHADER_MODEL_6_7 => naga::back::hlsl::ShaderModel::V6_7,
                _ => unreachable!(),
            }
        };

        let private_caps = super::PrivateCapabilities {
            instance_flags,
            heterogeneous_resource_heaps: options.ResourceHeapTier
                != Direct3D12::D3D12_RESOURCE_HEAP_TIER_1,
            memory_architecture: if features_architecture.UMA.as_bool() {
                super::MemoryArchitecture::Unified {
                    cache_coherent: features_architecture.CacheCoherentUMA.as_bool(),
                }
            } else {
                super::MemoryArchitecture::NonUnified
            },
            heap_create_not_zeroed,
            casting_fully_typed_format_supported,
            // See https://github.com/gfx-rs/wgpu/issues/3552
            suballocation_supported: !info.name.contains("Iris(R) Xe"),
            shader_model,
        };

        // Theoretically vram limited, but in practice 2^20 is the limit
        let tier3_practical_descriptor_limit = 1 << 20;

        let (full_heap_count, uav_count) = match options.ResourceBindingTier {
            Direct3D12::D3D12_RESOURCE_BINDING_TIER_1 => {
                let uav_count = match max_feature_level {
                    Direct3D::D3D_FEATURE_LEVEL_11_0 => 8,
                    _ => 64,
                };

                (
                    Direct3D12::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1,
                    uav_count,
                )
            }
            Direct3D12::D3D12_RESOURCE_BINDING_TIER_2 => (
                Direct3D12::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2,
                64,
            ),
            Direct3D12::D3D12_RESOURCE_BINDING_TIER_3 => (
                tier3_practical_descriptor_limit,
                tier3_practical_descriptor_limit,
            ),
            other => {
                log::warn!("Unknown resource binding tier {:?}", other);
                (
                    Direct3D12::D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_1,
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
            | wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgt::Features::TIMESTAMP_QUERY
            | wgt::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | wgt::Features::TEXTURE_COMPRESSION_BC
            | wgt::Features::TEXTURE_COMPRESSION_BC_SLICED_3D
            | wgt::Features::CLEAR_TEXTURE
            | wgt::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgt::Features::PUSH_CONSTANTS
            | wgt::Features::SHADER_PRIMITIVE_INDEX
            | wgt::Features::RG11B10UFLOAT_RENDERABLE
            | wgt::Features::DUAL_SOURCE_BLENDING
            | wgt::Features::TEXTURE_FORMAT_NV12;

        //TODO: in order to expose this, we need to run a compute shader
        // that extract the necessary statistics out of the D3D12 result.
        // Alternatively, we could allocate a buffer for the query set,
        // write the results there, and issue a bunch of copy commands.
        //| wgt::Features::PIPELINE_STATISTICS_QUERY

        if max_feature_level.0 >= Direct3D::D3D_FEATURE_LEVEL_11_1.0 {
            features |= wgt::Features::VERTEX_WRITABLE_STORAGE;
        }

        features.set(
            wgt::Features::CONSERVATIVE_RASTERIZATION,
            options.ConservativeRasterizationTier
                != Direct3D12::D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED,
        );

        features.set(
            wgt::Features::TEXTURE_BINDING_ARRAY
                | wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                | wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            shader_model >= naga::back::hlsl::ShaderModel::V5_1,
        );

        let bgra8unorm_storage_supported = {
            let mut bgra8unorm_info = Direct3D12::D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM,
                ..Default::default()
            };
            let hr = unsafe {
                device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_FORMAT_SUPPORT,
                    <*mut _>::cast(&mut bgra8unorm_info),
                    size_of_val(&bgra8unorm_info) as u32,
                )
            };
            hr.is_ok()
                && bgra8unorm_info
                    .Support2
                    .contains(Direct3D12::D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
        };
        features.set(
            wgt::Features::BGRA8UNORM_STORAGE,
            bgra8unorm_storage_supported,
        );

        let mut features1 = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let hr = unsafe {
            device.CheckFeatureSupport(
                Direct3D12::D3D12_FEATURE_D3D12_OPTIONS1,
                <*mut _>::cast(&mut features1),
                size_of_val(&features1) as u32,
            )
        };

        features.set(
            wgt::Features::SHADER_INT64,
            shader_model >= naga::back::hlsl::ShaderModel::V6_0
                && hr.is_ok()
                && features1.Int64ShaderOps.as_bool(),
        );

        features.set(
            wgt::Features::SUBGROUP,
            shader_model >= naga::back::hlsl::ShaderModel::V6_0
                && hr.is_ok()
                && features1.WaveOps.as_bool(),
        );

        let atomic_int64_on_typed_resource_supported = {
            let mut features9 = Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            unsafe {
                device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_D3D12_OPTIONS9,
                    <*mut _>::cast(&mut features9),
                    size_of_val(&features9) as u32,
                )
            }
            .is_ok()
                && features9.AtomicInt64OnGroupSharedSupported.as_bool()
                && features9.AtomicInt64OnTypedResourceSupported.as_bool()
        };
        features.set(
            wgt::Features::SHADER_INT64_ATOMIC_ALL_OPS | wgt::Features::SHADER_INT64_ATOMIC_MIN_MAX,
            atomic_int64_on_typed_resource_supported,
        );

        // float32-filterable should always be available on d3d12
        features.set(wgt::Features::FLOAT32_FILTERABLE, true);

        // TODO: Determine if IPresentationManager is supported
        let presentation_timer = auxil::dxgi::time::PresentationTimer::new_dxgi();

        let base = wgt::Limits::default();

        let mut downlevel = wgt::DownlevelCapabilities::default();
        // https://github.com/gfx-rs/wgpu/issues/2471
        downlevel.flags -=
            wgt::DownlevelFlags::VERTEX_AND_INSTANCE_INDEX_RESPECTS_RESPECTIVE_FIRST_VALUE_IN_INDIRECT_DRAW;

        // See https://learn.microsoft.com/en-us/windows/win32/direct3d12/hardware-feature-levels#feature-level-support
        let max_color_attachments = 8;
        // TODO: determine this programmatically if possible.
        // https://github.com/gpuweb/gpuweb/issues/2965#issuecomment-1361315447
        let max_color_attachment_bytes_per_sample = 64;

        Some(crate::ExposedAdapter {
            adapter: super::Adapter {
                raw: adapter,
                device,
                library: Arc::clone(library),
                private_caps,
                presentation_timer,
                workarounds,
                dxc_container,
            },
            info,
            features,
            capabilities: crate::Capabilities {
                limits: wgt::Limits {
                    max_texture_dimension_1d: Direct3D12::D3D12_REQ_TEXTURE1D_U_DIMENSION,
                    max_texture_dimension_2d: Direct3D12::D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
                        .min(Direct3D12::D3D12_REQ_TEXTURECUBE_DIMENSION),
                    max_texture_dimension_3d: Direct3D12::D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
                    max_texture_array_layers: Direct3D12::D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION,
                    max_bind_groups: crate::MAX_BIND_GROUPS as u32,
                    max_bindings_per_bind_group: 65535,
                    // dynamic offsets take a root constant, so we expose the minimum here
                    max_dynamic_uniform_buffers_per_pipeline_layout: base
                        .max_dynamic_uniform_buffers_per_pipeline_layout,
                    max_dynamic_storage_buffers_per_pipeline_layout: base
                        .max_dynamic_storage_buffers_per_pipeline_layout,
                    max_sampled_textures_per_shader_stage: match options.ResourceBindingTier {
                        Direct3D12::D3D12_RESOURCE_BINDING_TIER_1 => 128,
                        _ => full_heap_count,
                    },
                    max_samplers_per_shader_stage: match options.ResourceBindingTier {
                        Direct3D12::D3D12_RESOURCE_BINDING_TIER_1 => 16,
                        _ => Direct3D12::D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE,
                    },
                    // these both account towards `uav_count`, but we can't express the limit as as sum
                    // of the two, so we divide it by 4 to account for the worst case scenario
                    // (2 shader stages, with both using 16 storage textures and 16 storage buffers)
                    max_storage_buffers_per_shader_stage: uav_count / 4,
                    max_storage_textures_per_shader_stage: uav_count / 4,
                    max_uniform_buffers_per_shader_stage: full_heap_count,
                    max_uniform_buffer_binding_size:
                        Direct3D12::D3D12_REQ_CONSTANT_BUFFER_ELEMENT_COUNT * 16,
                    max_storage_buffer_binding_size: auxil::MAX_I32_BINDING_SIZE,
                    max_vertex_buffers: Direct3D12::D3D12_VS_INPUT_REGISTER_COUNT
                        .min(crate::MAX_VERTEX_BUFFERS as u32),
                    max_vertex_attributes: Direct3D12::D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT,
                    max_vertex_buffer_array_stride: Direct3D12::D3D12_SO_BUFFER_MAX_STRIDE_IN_BYTES,
                    min_subgroup_size: 4, // Not using `features1.WaveLaneCountMin` as it is unreliable
                    max_subgroup_size: 128,
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
                        Direct3D12::D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT,
                    min_storage_buffer_offset_alignment: 4,
                    max_inter_stage_shader_components: base.max_inter_stage_shader_components,
                    max_color_attachments,
                    max_color_attachment_bytes_per_sample,
                    max_compute_workgroup_storage_size: base.max_compute_workgroup_storage_size, //TODO?
                    max_compute_invocations_per_workgroup:
                        Direct3D12::D3D12_CS_4_X_THREAD_GROUP_MAX_THREADS_PER_GROUP,
                    max_compute_workgroup_size_x: Direct3D12::D3D12_CS_THREAD_GROUP_MAX_X,
                    max_compute_workgroup_size_y: Direct3D12::D3D12_CS_THREAD_GROUP_MAX_Y,
                    max_compute_workgroup_size_z: Direct3D12::D3D12_CS_THREAD_GROUP_MAX_Z,
                    max_compute_workgroups_per_dimension:
                        Direct3D12::D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION,
                    // Dx12 does not expose a maximum buffer size in the API.
                    // This limit is chosen to avoid potential issues with drivers should they internally
                    // store buffer sizes using 32 bit ints (a situation we have already encountered with vulkan).
                    max_buffer_size: i32::MAX as u64,
                    max_non_sampler_bindings: 1_000_000,
                },
                alignments: crate::Alignments {
                    buffer_copy_offset: wgt::BufferSize::new(
                        Direct3D12::D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as u64,
                    )
                    .unwrap(),
                    buffer_copy_pitch: wgt::BufferSize::new(
                        Direct3D12::D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as u64,
                    )
                    .unwrap(),
                    // Direct3D correctly bounds-checks all array accesses:
                    // https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#18.6.8.2%20Device%20Memory%20Reads
                    uniform_bounds_check_alignment: wgt::BufferSize::new(1).unwrap(),
                },
                downlevel,
            },
        })
    }
}

impl crate::Adapter for super::Adapter {
    type A = super::Api;

    unsafe fn open(
        &self,
        _features: wgt::Features,
        limits: &wgt::Limits,
        memory_hints: &wgt::MemoryHints,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        let queue: Direct3D12::ID3D12CommandQueue = {
            profiling::scope!("ID3D12Device::CreateCommandQueue");
            unsafe {
                self.device
                    .CreateCommandQueue(&Direct3D12::D3D12_COMMAND_QUEUE_DESC {
                        Type: Direct3D12::D3D12_COMMAND_LIST_TYPE_DIRECT,
                        Priority: Direct3D12::D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
                        Flags: Direct3D12::D3D12_COMMAND_QUEUE_FLAG_NONE,
                        NodeMask: 0,
                    })
            }
            .into_device_result("Queue creation")?
        };

        let device = super::Device::new(
            self.device.clone(),
            queue.clone(),
            limits,
            memory_hints,
            self.private_caps,
            &self.library,
            self.dxc_container.clone(),
        )?;
        Ok(crate::OpenDevice {
            device,
            queue: super::Queue {
                raw: queue,
                temp_lists: Mutex::new(Vec::new()),
            },
        })
    }

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

        let mut data = Direct3D12::D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: raw_format,
            ..Default::default()
        };
        unsafe {
            self.device.CheckFeatureSupport(
                Direct3D12::D3D12_FEATURE_FORMAT_SUPPORT,
                <*mut _>::cast(&mut data),
                size_of_val(&data) as u32,
            )
        }
        .unwrap();

        // Because we use a different format for SRV and UAV views of depth textures, we need to check
        // the features that use SRV/UAVs using the no-depth format.
        let mut data_srv_uav = Direct3D12::D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: srv_uav_format,
            Support1: Direct3D12::D3D12_FORMAT_SUPPORT1_NONE,
            Support2: Direct3D12::D3D12_FORMAT_SUPPORT2_NONE,
        };
        if raw_format != srv_uav_format {
            // Only-recheck if we're using a different format
            unsafe {
                self.device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_FORMAT_SUPPORT,
                    ptr::addr_of_mut!(data_srv_uav).cast(),
                    size_of::<Direct3D12::D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32,
                )
            }
            .unwrap();
        } else {
            // Same format, just copy over.
            data_srv_uav = data;
        }

        let mut caps = Tfc::COPY_SRC | Tfc::COPY_DST;
        // Cannot use the contains() helper, and windows-rs doesn't provide a .intersect() helper
        let is_texture = (data.Support1
            & (Direct3D12::D3D12_FORMAT_SUPPORT1_TEXTURE1D
                | Direct3D12::D3D12_FORMAT_SUPPORT1_TEXTURE2D
                | Direct3D12::D3D12_FORMAT_SUPPORT1_TEXTURE3D
                | Direct3D12::D3D12_FORMAT_SUPPORT1_TEXTURECUBE))
            .0
            != 0;
        // SRVs use srv_uav_format
        caps.set(
            Tfc::SAMPLED,
            is_texture
                && data_srv_uav
                    .Support1
                    .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
        );
        caps.set(
            Tfc::SAMPLED_LINEAR,
            data_srv_uav
                .Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE),
        );
        caps.set(
            Tfc::COLOR_ATTACHMENT,
            data.Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
        );
        caps.set(
            Tfc::COLOR_ATTACHMENT_BLEND,
            data.Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_BLENDABLE),
        );
        caps.set(
            Tfc::DEPTH_STENCIL_ATTACHMENT,
            data.Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL),
        );
        // UAVs use srv_uav_format
        caps.set(
            Tfc::STORAGE,
            data_srv_uav
                .Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW),
        );
        caps.set(
            Tfc::STORAGE_READ_WRITE,
            data_srv_uav
                .Support2
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD),
        );

        // We load via UAV/SRV so use srv_uav_format
        let no_msaa_load = caps.contains(Tfc::SAMPLED)
            && !data_srv_uav
                .Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD);

        let no_msaa_target = (data.Support1
            & (Direct3D12::D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                | Direct3D12::D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL))
            .0
            != 0
            && !data
                .Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET);

        caps.set(
            Tfc::MULTISAMPLE_RESOLVE,
            data.Support1
                .contains(Direct3D12::D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE),
        );

        let mut ms_levels = Direct3D12::D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: raw_format,
            SampleCount: 0,
            Flags: Direct3D12::D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
            NumQualityLevels: 0,
        };

        let mut set_sample_count = |sc: u32, tfc: Tfc| {
            ms_levels.SampleCount = sc;

            if unsafe {
                self.device.CheckFeatureSupport(
                    Direct3D12::D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                    <*mut _>::cast(&mut ms_levels),
                    size_of_val(&ms_levels) as u32,
                )
            }
            .is_ok()
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
                    let mut rect = Default::default();
                    if unsafe { WindowsAndMessaging::GetClientRect(wnd_handle, &mut rect) }.is_ok()
                    {
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
                SurfaceTarget::Visual(_)
                | SurfaceTarget::SurfaceHandle(_)
                | SurfaceTarget::SwapChainPanel(_) => None,
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
            // See https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-idxgidevice1-setmaximumframelatency
            maximum_frame_latency: 1..=16,
            current_extent,
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
