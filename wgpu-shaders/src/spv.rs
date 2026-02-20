use alloc::borrow::Cow;
#[cfg(feature = "naga-dep")]
use naga::back::spv::Options;

/// Global settings for a device. Not specialized to a shader.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpvCompileOptionsDesc {
    /// SPIR-V language version (e.g. 1.6)
    pub lang_version: (u8, u8),
    /// Set of features enabled on the device
    pub features: wgt::Features,
    /// Task dispatch limits
    pub task_dispatch_limits: wst::TaskDispatchLimits,
    /// Downlevel flags
    pub downlevel_flags: wgt::DownlevelFlags,
    /// Instance flags. Used for debugging
    pub instance_flags: wgt::InstanceFlags,
    /// True if VK_KHR_shader_non_semantic_infoshader_non_semantic_info is enabled or api version >= 1.3.
    ///
    /// Used for certain debug instructions
    pub shader_non_semantic_info: bool,
    /// Pass true if this might run on qualcomm. Used for vendor-specific workarounds.
    ///
    /// Used for certain debug instructions
    pub vendor_could_be_qualcomm: bool,
    /// Enable support for integer dot products
    pub shader_integer_dot_product: bool,
    /// Enable support for 8 bit integers
    pub shader_int8: bool,
    /// Whether the device supports driver-implemented memory zero initialization
    pub native_zero_init: bool,
    /// Whether or not the device supports f16 in shader IO. Otherwise this will be polyfilled
    pub shader_input_output_16: bool,
    /// If false, buffer accesses will be bounds checked manually
    pub robust_buffer_access2: bool,
    /// If false, image accesses will be bounds checked manually
    pub robust_image_access: bool,
}
impl SpvCompileOptionsDesc {
    /// These are the absolutely lowest possible options. It is recommended
    /// to look over the options and tweak a few yourself, as this may result
    /// in larger and slower shaders that must emulate some features and
    /// doesn't even support others.
    pub fn maximum_compat() -> Self {
        Self {
            lang_version: (1, 0),
            features: wgt::Features::empty(),
            task_dispatch_limits: wst::TaskDispatchLimits {
                max_mesh_workgroups_per_dim: 64,
                max_mesh_workgroups_total: 1024,
            },
            downlevel_flags: wgt::DownlevelFlags::empty(),
            instance_flags: wgt::InstanceFlags::empty(),
            shader_non_semantic_info: false,
            vendor_could_be_qualcomm: true,
            shader_integer_dot_product: false,
            shader_int8: false,
            native_zero_init: false,
            shader_input_output_16: false,
            robust_buffer_access2: false,
            robust_image_access: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpvCompileOptions<'a> {
    #[cfg(feature = "naga-dep")]
    pub(crate) options: Options<'a>,
    #[cfg(not(feature = "naga-dep"))]
    pub(crate) _p: core::marker::PhantomData<&'a ()>,
}
impl SpvCompileOptions<'_> {
    pub fn new(desc: SpvCompileOptionsDesc) -> Self {
        #[cfg(feature = "naga-dep")]
        {
            use naga::back::spv;

            // The following capabilities are always available
            // see https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap52.html#spirvenv-capabilities
            let mut capabilities = vec![
                spv::Capability::Shader,
                spv::Capability::Matrix,
                spv::Capability::Sampled1D,
                spv::Capability::Image1D,
                spv::Capability::ImageQuery,
                spv::Capability::DerivativeControl,
                spv::Capability::StorageImageExtendedFormats,
            ];

            if desc
                .downlevel_flags
                .contains(wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES)
            {
                capabilities.push(spv::Capability::SampledCubeArray);
            }

            if desc
                .downlevel_flags
                .contains(wgt::DownlevelFlags::MULTISAMPLED_SHADING)
            {
                capabilities.push(spv::Capability::SampleRateShading);
            }

            if desc.features.contains(wgt::Features::MULTIVIEW) {
                capabilities.push(spv::Capability::MultiView);
            }

            if desc
                .features
                .contains(wgt::Features::SHADER_PRIMITIVE_INDEX)
            {
                capabilities.push(spv::Capability::Geometry);
            }

            if desc
                .features
                .intersects(wgt::Features::SUBGROUP | wgt::Features::SUBGROUP_VERTEX)
            {
                capabilities.push(spv::Capability::GroupNonUniform);
                capabilities.push(spv::Capability::GroupNonUniformVote);
                capabilities.push(spv::Capability::GroupNonUniformArithmetic);
                capabilities.push(spv::Capability::GroupNonUniformBallot);
                capabilities.push(spv::Capability::GroupNonUniformShuffle);
                capabilities.push(spv::Capability::GroupNonUniformShuffleRelative);
                capabilities.push(spv::Capability::GroupNonUniformQuad);
            }

            if desc.features.intersects(
                wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                    | wgt::Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
                    | wgt::Features::UNIFORM_BUFFER_BINDING_ARRAYS,
            ) {
                capabilities.push(spv::Capability::ShaderNonUniform);
            }
            if desc.features.contains(wgt::Features::BGRA8UNORM_STORAGE) {
                capabilities.push(spv::Capability::StorageImageWriteWithoutFormat);
            }

            if desc
                .features
                .contains(wgt::Features::EXPERIMENTAL_RAY_QUERY)
            {
                capabilities.push(spv::Capability::RayQueryKHR);
            }

            if desc.features.contains(wgt::Features::SHADER_INT64) {
                capabilities.push(spv::Capability::Int64);
            }

            if desc.features.contains(wgt::Features::SHADER_F16) {
                capabilities.push(spv::Capability::Float16);
            }

            if desc.features.intersects(
                wgt::Features::SHADER_INT64_ATOMIC_ALL_OPS
                    | wgt::Features::SHADER_INT64_ATOMIC_MIN_MAX
                    | wgt::Features::TEXTURE_INT64_ATOMIC,
            ) {
                capabilities.push(spv::Capability::Int64Atomics);
            }

            if desc
                .features
                .intersects(wgt::Features::TEXTURE_INT64_ATOMIC)
            {
                capabilities.push(spv::Capability::Int64ImageEXT);
            }

            if desc.features.contains(wgt::Features::SHADER_FLOAT32_ATOMIC) {
                capabilities.push(spv::Capability::AtomicFloat32AddEXT);
            }

            if desc.features.contains(wgt::Features::CLIP_DISTANCES) {
                capabilities.push(spv::Capability::ClipDistance);
            }

            // Vulkan bundles both barycentrics and per-vertex attributes under the same feature.
            if desc
                .features
                .intersects(wgt::Features::SHADER_BARYCENTRICS | wgt::Features::SHADER_PER_VERTEX)
            {
                capabilities.push(spv::Capability::FragmentBarycentricKHR);
            }

            if desc.features.contains(wgt::Features::SHADER_DRAW_INDEX) {
                capabilities.push(spv::Capability::DrawParameters);
            }

            let mut flags = spv::WriterFlags::empty();
            flags.set(
                spv::WriterFlags::DEBUG,
                desc.instance_flags.contains(wgt::InstanceFlags::DEBUG),
            );
            flags.set(
                spv::WriterFlags::LABEL_VARYINGS,
                !desc.vendor_could_be_qualcomm,
            );
            flags.set(
                spv::WriterFlags::FORCE_POINT_SIZE,
                //Note: we could technically disable this when we are compiling separate entry points,
                // and we know exactly that the primitive topology is not `PointList`.
                // But this requires cloning the `spv::Options` struct, which has heap allocations.
                true, // could check `super::Workarounds::SEPARATE_ENTRY_POINTS`
            );
            flags.set(
                spv::WriterFlags::PRINT_ON_RAY_QUERY_INITIALIZATION_FAIL,
                desc.instance_flags.contains(wgt::InstanceFlags::DEBUG)
                    && desc.shader_non_semantic_info,
            );
            if desc
                .features
                .contains(wgt::Features::EXPERIMENTAL_RAY_QUERY)
            {
                capabilities.push(spv::Capability::RayQueryKHR);
            }
            if desc
                .features
                .contains(wgt::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)
            {
                capabilities.push(spv::Capability::RayQueryPositionFetchKHR)
            }
            if desc
                .features
                .contains(wgt::Features::EXPERIMENTAL_MESH_SHADER)
            {
                capabilities.push(spv::Capability::MeshShadingEXT);
            }
            if desc
                .features
                .contains(wgt::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            {
                capabilities.push(spv::Capability::CooperativeMatrixKHR);
                // TODO: expose this more generally
                capabilities.push(spv::Capability::VulkanMemoryModel);
            }
            if desc.shader_integer_dot_product {
                // See <https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_integer_dot_product.html#_new_spir_v_capabilities>.
                capabilities.extend(&[
                    spv::Capability::DotProductInputAllKHR,
                    spv::Capability::DotProductInput4x8BitKHR,
                    spv::Capability::DotProductInput4x8BitPackedKHR,
                    spv::Capability::DotProductKHR,
                ]);
            }
            if desc.shader_int8 {
                // See <https://registry.khronos.org/vulkan/specs/latest/man/html/VkPhysicalDeviceShaderFloat16Int8Features.html#extension-features-shaderInt8>.
                capabilities.extend(&[spv::Capability::Int8]);
            }
            Self {
                #[cfg(feature = "naga-dep")]
                options: Options {
                    lang_version: desc.lang_version,
                    flags,
                    capabilities: Some(capabilities.iter().cloned().collect()),
                    bounds_check_policies: naga::proc::BoundsCheckPolicies {
                        index: naga::proc::BoundsCheckPolicy::Restrict,
                        buffer: if desc.robust_buffer_access2 {
                            naga::proc::BoundsCheckPolicy::Unchecked
                        } else {
                            naga::proc::BoundsCheckPolicy::Restrict
                        },
                        image_load: if desc.robust_image_access {
                            naga::proc::BoundsCheckPolicy::Unchecked
                        } else {
                            naga::proc::BoundsCheckPolicy::Restrict
                        },
                        // TODO: support bounds checks on binding arrays
                        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                    },
                    zero_initialize_workgroup_memory: if desc.native_zero_init {
                        spv::ZeroInitializeWorkgroupMemoryMode::Native
                    } else {
                        spv::ZeroInitializeWorkgroupMemoryMode::Polyfill
                    },
                    force_loop_bounding: true,
                    ray_query_initialization_tracking: true,
                    use_storage_input_output_16: desc.features.contains(wgt::Features::SHADER_F16)
                        && desc.shader_input_output_16,
                    fake_missing_bindings: false,
                    // We need to build this separately for each invocation, so just default it out here
                    binding_map: alloc::collections::BTreeMap::default(),
                    debug_info: None,
                    task_dispatch_limits: Some(desc.task_dispatch_limits),
                    mesh_shader_primitive_indices_clamp: true,
                },
            }
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            Self {
                _p: Default::default(),
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpvShaderDesc<'a> {
    pub options: Cow<'a, SpvCompileOptions<'a>>,
    pub runtime_checks: wgt::ShaderRuntimeChecks,
    pub entry_point: Option<(String, wst::ShaderStage)>,
    pub binding_map: &'a wst::spv::BindingMap,
    pub zero_initialize_workgroup_memory: bool,
    pub constants: &'a wst::PipelineConstants,
}

impl super::NagaShader {
    pub fn compile_spv(
        &self,
        desc: SpvShaderDesc,
    ) -> Result<Vec<u32>, crate::ShaderCompilationError> {
        #[cfg(feature = "naga-dep")]
        {
            let needs_temp_options = !desc.runtime_checks.bounds_checks
                || !desc.runtime_checks.force_loop_bounding
                || !desc.runtime_checks.ray_query_initialization_tracking
                || !desc.binding_map.is_empty()
                || self.debug_source.is_some()
                || !desc.zero_initialize_workgroup_memory
                || !desc.runtime_checks.task_shader_dispatch_tracking
                || !desc.runtime_checks.mesh_shader_primitive_indices_clamp;

            let options = if needs_temp_options {
                let mut temp_options: SpvCompileOptions<'_> = desc.options.into_owned();
                let temp_naga_options = &mut temp_options.options;
                if !desc.runtime_checks.bounds_checks {
                    temp_naga_options.bounds_check_policies = naga::proc::BoundsCheckPolicies {
                        index: naga::proc::BoundsCheckPolicy::Unchecked,
                        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
                        image_load: naga::proc::BoundsCheckPolicy::Unchecked,
                        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                    };
                }
                if !desc.runtime_checks.force_loop_bounding {
                    temp_naga_options.force_loop_bounding = false;
                }
                if !desc.runtime_checks.ray_query_initialization_tracking {
                    temp_naga_options.ray_query_initialization_tracking = false;
                }
                if !desc.binding_map.is_empty() {
                    temp_naga_options.binding_map = desc.binding_map.clone();
                }

                if let Some(ref debug) = self.debug_source {
                    temp_naga_options.debug_info = Some(naga::back::spv::DebugInfo {
                        source_code: &debug.source_code,
                        file_name: debug.file_name.as_ref(),
                        language: naga::back::spv::SourceLanguage::WGSL,
                    })
                }
                if !desc.zero_initialize_workgroup_memory {
                    temp_naga_options.zero_initialize_workgroup_memory =
                        naga::back::spv::ZeroInitializeWorkgroupMemoryMode::None;
                }
                if !desc.runtime_checks.task_shader_dispatch_tracking {
                    temp_naga_options.task_dispatch_limits = None;
                }
                temp_naga_options.mesh_shader_primitive_indices_clamp =
                    desc.runtime_checks.mesh_shader_primitive_indices_clamp;
                Cow::Owned(temp_options)
            } else {
                Cow::Borrowed(&*desc.options)
            };

            let (module, info): (Cow<naga::Module>, Cow<naga::valid::ModuleInfo>) =
                if desc.entry_point.is_some() {
                    naga::back::pipeline_constants::process_overrides(
                        &self.module,
                        &self.info,
                        desc.entry_point
                            .as_ref()
                            .map(|(name, stage)| (*stage, name.as_str())),
                        desc.constants,
                    )
                    .map_err(|e| crate::ShaderCompilationError::PipelineConstants(format!("{e}")))?
                } else {
                    (Cow::Borrowed(&self.module), Cow::Borrowed(&self.info))
                };

            let pipeline_options =
                desc.entry_point
                    .map(|(name, stage)| naga::back::spv::PipelineOptions {
                        entry_point: name,
                        shader_stage: stage,
                    });
            {
                profiling::scope!("naga::spv::write_vec");
                naga::back::spv::write_vec(
                    &module,
                    &info,
                    &options.options,
                    pipeline_options.as_ref(),
                )
            }
            .map_err(|e| crate::ShaderCompilationError::Linkage(format!("{e}")))
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            unreachable!()
        }
    }
}
