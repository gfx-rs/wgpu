use std::num::NonZeroU64;

use winapi::um::{d3d11, d3dcommon};

impl crate::Adapter<super::Api> for super::Adapter {
    unsafe fn open(
        &self,
        features: wgt::Features,
        limits: &wgt::Limits,
    ) -> Result<crate::OpenDevice<super::Api>, crate::DeviceError> {
        todo!()
    }

    unsafe fn texture_format_capabilities(
        &self,
        format: wgt::TextureFormat,
    ) -> crate::TextureFormatCapabilities {
        todo!()
    }

    unsafe fn surface_capabilities(
        &self,
        surface: &super::Surface,
    ) -> Option<crate::SurfaceCapabilities> {
        todo!()
    }
}

impl super::Adapter {
    pub(super) fn expose(
        instance: &super::library::D3D11Lib,
        adapter: native::DxgiAdapter,
    ) -> Option<crate::ExposedAdapter<super::Api>> {
        use d3dcommon::{
            D3D_FEATURE_LEVEL_10_0 as FL10_0, D3D_FEATURE_LEVEL_10_1 as FL10_1,
            D3D_FEATURE_LEVEL_11_0 as FL11_0, D3D_FEATURE_LEVEL_11_1 as FL11_1,
            D3D_FEATURE_LEVEL_9_1 as FL9_1, D3D_FEATURE_LEVEL_9_2 as FL9_2,
            D3D_FEATURE_LEVEL_9_3 as FL9_3,
        };

        let (device, feature_level) = instance.create_device(adapter)?;

        //
        // Query Features from d3d11
        //

        let d3d9_features = unsafe {
            device.check_feature_support::<d3d11::D3D11_FEATURE_DATA_D3D9_OPTIONS1>(
                d3d11::D3D11_FEATURE_D3D9_OPTIONS1,
            )
        };

        let d3d10_features = unsafe {
            device.check_feature_support::<d3d11::D3D11_FEATURE_DATA_D3D10_X_HARDWARE_OPTIONS>(
                d3d11::D3D11_FEATURE_D3D10_X_HARDWARE_OPTIONS,
            )
        };

        let d3d11_features = unsafe {
            device.check_feature_support::<d3d11::D3D11_FEATURE_DATA_D3D11_OPTIONS>(
                d3d11::D3D11_FEATURE_D3D11_OPTIONS,
            )
        };

        let d3d11_features1 = unsafe {
            device.check_feature_support::<d3d11::D3D11_FEATURE_DATA_D3D11_OPTIONS1>(
                d3d11::D3D11_FEATURE_D3D11_OPTIONS1,
            )
        };

        let d3d11_features2 = unsafe {
            device.check_feature_support::<d3d11::D3D11_FEATURE_DATA_D3D11_OPTIONS2>(
                d3d11::D3D11_FEATURE_D3D11_OPTIONS2,
            )
        };

        let d3d11_features3 = unsafe {
            device.check_feature_support::<d3d11::D3D11_FEATURE_DATA_D3D11_OPTIONS3>(
                d3d11::D3D11_FEATURE_D3D11_OPTIONS3,
            )
        };

        //
        // Fill out features and downlevel features
        //
        // TODO(cwfitzgerald): Needed downlevel features: 3D dispatch

        let mut features = wgt::Features::DEPTH_CLIP_CONTROL
            | wgt::Features::PUSH_CONSTANTS
            | wgt::Features::POLYGON_MODE_LINE
            | wgt::Features::CLEAR_TEXTURE
            | wgt::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgt::Features::ADDRESS_MODE_CLAMP_TO_ZERO;
        let mut downlevel =
            wgt::DownlevelFlags::BASE_VERTEX | wgt::DownlevelFlags::READ_ONLY_DEPTH_STENCIL;

        // Features from queries
        downlevel.set(
            wgt::DownlevelFlags::NON_POWER_OF_TWO_MIPMAPPED_TEXTURES,
            d3d9_features.FullNonPow2TextureSupported == 1,
        );
        downlevel.set(
            wgt::DownlevelFlags::COMPUTE_SHADERS,
            d3d10_features.ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x == 1,
        );

        // Features from feature level
        if feature_level >= FL9_2 {
            downlevel |= wgt::DownlevelFlags::INDEPENDENT_BLEND;
            // formally FL9_1 supports aniso 2, but we don't support that level of distinction
            downlevel |= wgt::DownlevelFlags::ANISOTROPIC_FILTERING;
        }

        if feature_level >= FL9_3 {
            downlevel |= wgt::DownlevelFlags::COMPARISON_SAMPLERS;
        }

        if feature_level >= FL10_0 {
            downlevel |= wgt::DownlevelFlags::INDEPENDENT_BLEND;
            downlevel |= wgt::DownlevelFlags::FRAGMENT_STORAGE;
            downlevel |= wgt::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE;
            features |= wgt::Features::DEPTH_CLIP_CONTROL;
            features |= wgt::Features::TIMESTAMP_QUERY;
            features |= wgt::Features::PIPELINE_STATISTICS_QUERY;
        }

        if feature_level >= FL10_1 {
            downlevel |= wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES;
        }

        if feature_level >= FL11_0 {
            downlevel |= wgt::DownlevelFlags::INDIRECT_EXECUTION;
            downlevel |= wgt::DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT;
            features |= wgt::Features::TEXTURE_COMPRESSION_BC;
        }

        if feature_level >= FL11_1 {
            downlevel |= wgt::DownlevelFlags::VERTEX_STORAGE;
        }

        //
        // Fill out limits and alignments
        //

        let max_texture_dimension_2d = match feature_level {
            FL9_1 | FL9_2 => 2048,
            FL9_3 => 4096,
            FL10_0 | FL10_1 => 8192,
            _ => d3d11::D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION,
        };

        let max_texture_dimension_3d = match feature_level {
            FL9_1..=FL9_3 => 256,
            _ => d3d11::D3D11_REQ_TEXTURE3D_U_V_OR_W_DIMENSION,
        };
        let max_vertex_buffers = match feature_level {
            FL9_1..=FL9_3 => 16,
            _ => 32,
        };
        let max_compute_workgroup_storage_size = match feature_level {
            FL9_1..=FL9_3 => 0,
            FL10_0 | FL10_1 => 4096 * 4, // This doesn't have an equiv SM4 constant :\
            _ => d3d11::D3D11_CS_TGSM_REGISTER_COUNT * 4,
        };
        let max_workgroup_size_xy = match feature_level {
            FL9_1..=FL9_3 => 0,
            FL10_0 | FL10_1 => d3d11::D3D11_CS_4_X_THREAD_GROUP_MAX_X,
            _ => d3d11::D3D11_CS_THREAD_GROUP_MAX_X,
        };
        let max_workgroup_size_z = match feature_level {
            FL9_1..=FL9_3 => 0,
            FL10_0 | FL10_1 => 1,
            _ => d3d11::D3D11_CS_THREAD_GROUP_MAX_Z,
        };
        // let max_workgroup_count_z = match feature_level {
        //     FL9_1..=FL9_3 => 0,
        //     FL10_0 | FL10_1 => 1,
        //     _ => d3d11::D3D11_CS_THREAD_GROUP_MAX_Z,
        // };

        let max_sampled_textures = d3d11::D3D11_COMMONSHADER_INPUT_RESOURCE_REGISTER_COUNT;
        let max_samplers = d3d11::D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
        let max_constant_buffers = d3d11::D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT - 1;
        let max_uavs = if device.as_device1().is_some() {
            d3d11::D3D11_1_UAV_SLOT_COUNT
        } else {
            d3d11::D3D11_PS_CS_UAV_REGISTER_COUNT
        };
        let max_output_registers = d3d11::D3D11_VS_OUTPUT_REGISTER_COMPONENTS;
        let max_compute_invocations_per_workgroup =
            d3d11::D3D11_CS_THREAD_GROUP_MAX_THREADS_PER_GROUP;
        let max_compute_workgroups_per_dimension =
            d3d11::D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

        let limits = wgt::Limits {
            max_texture_dimension_1d: max_texture_dimension_2d,
            max_texture_dimension_2d,
            max_texture_dimension_3d,
            max_texture_array_layers: max_texture_dimension_3d,
            max_bind_groups: u32::MAX,
            max_dynamic_uniform_buffers_per_pipeline_layout: max_constant_buffers,
            max_dynamic_storage_buffers_per_pipeline_layout: 0,
            max_sampled_textures_per_shader_stage: max_sampled_textures,
            max_samplers_per_shader_stage: max_samplers,
            max_storage_buffers_per_shader_stage: max_uavs,
            max_storage_textures_per_shader_stage: max_uavs,
            max_uniform_buffers_per_shader_stage: max_constant_buffers,
            max_uniform_buffer_binding_size: 1 << 16,
            max_storage_buffer_binding_size: u32::MAX,
            max_vertex_buffers,
            max_vertex_attributes: max_vertex_buffers,
            max_vertex_buffer_array_stride: u32::MAX,
            max_push_constant_size: 1 << 16,
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 1,
            max_inter_stage_shader_components: max_output_registers,
            max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: max_workgroup_size_xy,
            max_compute_workgroup_size_y: max_workgroup_size_xy,
            max_compute_workgroup_size_z: max_workgroup_size_z,
            max_compute_workgroups_per_dimension,
            // D3D11_BUFFER_DESC represents the buffer size as a 32 bit int.
            max_buffer_size: u32::MAX as u64,
        };

        //
        // Other capabilities
        //

        let shader_model = match feature_level {
            FL9_1..=FL9_3 => wgt::ShaderModel::Sm2,
            FL10_0 | FL10_1 => wgt::ShaderModel::Sm4,
            _ => wgt::ShaderModel::Sm5,
        };

        let device_info = wgt::AdapterInfo {
            name: String::new(),
            vendor: 0,
            device: 0,
            device_type: match d3d11_features2.UnifiedMemoryArchitecture {
                0 => wgt::DeviceType::DiscreteGpu,
                1 => wgt::DeviceType::IntegratedGpu,
                _ => unreachable!(),
            },
            backend: wgt::Backend::Dx11,
        };

        //
        // Build up the structs
        //

        let api_adapter = super::Adapter { device };

        let alignments = crate::Alignments {
            buffer_copy_offset: NonZeroU64::new(1).unwrap(), // todo
            buffer_copy_pitch: NonZeroU64::new(1).unwrap(),  // todo
        };

        let capabilities = crate::Capabilities {
            limits,
            alignments,
            downlevel: wgt::DownlevelCapabilities {
                flags: downlevel,
                limits: wgt::DownlevelLimits {},
                shader_model,
            },
        };

        Some(crate::ExposedAdapter {
            adapter: api_adapter,
            info: device_info,
            features,
            capabilities,
        })
    }
}
