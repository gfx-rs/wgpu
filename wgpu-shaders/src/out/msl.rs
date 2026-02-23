use alloc::borrow::Cow;
#[cfg(feature = "naga-dep")]
use naga::back::msl::Options;

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MslCompileOptionsDesc {
    pub lang_version: (u8, u8),
}
impl MslCompileOptionsDesc {
    /// These are the absolutely lowest possible options. It is recommended
    /// to look over the options and tweak a few yourself, as this may result
    /// in larger and slower shaders that must emulate some features and
    /// doesn't even support others.
    pub fn maximum_compat() -> Self {
        Self {
            lang_version: (1, 0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MslCompileOptions {
    #[cfg(feature = "naga-dep")]
    pub(crate) options: Options,
}
impl MslCompileOptions {
    pub fn new(desc: MslCompileOptionsDesc) -> Self {
        #[cfg(feature = "naga-dep")]
        {
            Self {
                // TODO
                options: naga::back::msl::Options {
                    lang_version: desc.lang_version,
                    inline_samplers: Default::default(),
                    spirv_cross_compatibility: false,
                    fake_missing_bindings: false,
                    per_entry_point_map: Default::default(),
                    bounds_check_policies: Default::default(),
                    zero_initialize_workgroup_memory: true,
                    force_loop_bounding: true,
                },
            }
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            Self {}
        }
    }
}

pub struct MslEntryPointCompileResult {
    pub compiled_name: String,
    pub workgroup_size: [u32; 3],
    pub wg_memory_sizes: Vec<u32>,
    pub sized_bindings: Vec<wst::ResourceBinding>,
    pub immutable_buffer_mask: usize,
}

pub struct MslCompileResult {
    pub shader: String,
    pub entry_points: Vec<MslEntryPointCompileResult>,
}

#[derive(Clone, Debug)]
pub struct MslShaderDesc<'a> {
    pub options: Cow<'a, MslCompileOptions>,
    pub runtime_checks: wgt::ShaderRuntimeChecks,
    pub entry_point: Option<(&'a str, wst::ShaderStage)>,
    pub resources: &'a wst::msl::EntryPointResources,
    pub constants: &'a wst::PipelineConstants,
    pub vertex_buffer_mappings: &'a [wst::msl::VertexBufferMapping],
    pub zero_init_memory: bool,
    pub is_point_primitive: bool,
}

impl crate::NagaShader {
    pub fn compile_msl(
        &self,
        desc: MslShaderDesc,
    ) -> Result<MslCompileResult, super::ShaderCompilationError> {
        #[cfg(feature = "naga-dep")]
        {
            let (module, module_info) = naga::back::pipeline_constants::process_overrides(
                &self.module,
                &self.info,
                desc.entry_point.map(|e| (e.1, e.0)),
                desc.constants,
            )
            .map_err(|e| super::ShaderCompilationError::PipelineConstants(format!("MSL: {e:?}")))?;

            let ep_resources = desc.resources;

            let bounds_check_policy = if desc.runtime_checks.bounds_checks {
                naga::proc::BoundsCheckPolicy::Restrict
            } else {
                naga::proc::BoundsCheckPolicy::Unchecked
            };

            let per_entry_point_map = if let Some((name, _)) = desc.entry_point {
                naga::back::msl::EntryPointResourceMap::from([(
                    name.to_owned(),
                    ep_resources.clone(),
                )])
            } else {
                Default::default()
            };

            let options = naga::back::msl::Options {
                inline_samplers: Default::default(),
                spirv_cross_compatibility: false,
                fake_missing_bindings: false,
                per_entry_point_map,
                bounds_check_policies: naga::proc::BoundsCheckPolicies {
                    index: bounds_check_policy,
                    buffer: bounds_check_policy,
                    image_load: bounds_check_policy,
                    // TODO: support bounds checks on binding arrays
                    binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                },
                zero_initialize_workgroup_memory: desc.zero_init_memory,
                force_loop_bounding: desc.runtime_checks.force_loop_bounding,
                ..desc.options.options
            };

            let pipeline_options = naga::back::msl::PipelineOptions {
                entry_point: desc.entry_point.map(|e| {
                    (
                        e.1,
                        e.0.to_owned(),
                        if desc.is_point_primitive {
                            naga::PrimitiveTopology::Points
                        } else {
                            naga::PrimitiveTopology::Triangles
                        },
                    )
                }),
                vertex_pulling_transform: true,
                vertex_buffer_mappings: desc.vertex_buffer_mappings.to_vec(),
            };

            let (shader, info) =
                naga::back::msl::write_string(&module, &module_info, &options, &pipeline_options)
                    .map_err(|e| super::ShaderCompilationError::Linkage(format!("MSL: {e:?}")))?;

            for (i, e) in info.entry_point_names.iter().enumerate() {
                if let Err(e) = e {
                    return Err(super::ShaderCompilationError::Linkage(format!(
                        "Error in entry point {}: {e}",
                        module.entry_points[i].name
                    )));
                }
            }

            let ep_range = if let Some((ep_name, ep_stage)) = desc.entry_point {
                let ep_index = module
                    .entry_points
                    .iter()
                    .position(|ep| ep.name == ep_name && ep.stage == ep_stage)
                    .expect("entry point not found in module");
                ep_index..ep_index + 1
            } else {
                0..module.entry_points.len()
            };

            let mut entry_points = Vec::new();
            for ep_index in ep_range {
                let ep = &module.entry_points[ep_index];
                let ep_info = module_info.get_entry_point(ep_index);

                let mut wg_memory_sizes = Vec::new();
                let mut immutable_buffer_mask = 0;
                let mut sized_bindings = Vec::new();

                for (handle, var) in module.global_variables.iter() {
                    if ep_info[handle].is_empty() {
                        continue;
                    }
                    match var.space {
                        naga::AddressSpace::WorkGroup => {
                            wg_memory_sizes.push(module.types[var.ty].inner.size(module.to_ctx()));
                        }
                        naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {
                            let br = match var.binding {
                                Some(br) => br,
                                None => continue,
                            };
                            let storage_access_store = match var.space {
                                naga::AddressSpace::Storage { access } => {
                                    access.contains(naga::StorageAccess::STORE)
                                }
                                _ => false,
                            };

                            if !ep_info[handle].is_empty() && !storage_access_store {
                                let slot = ep_resources.resources[&br].buffer.unwrap();
                                immutable_buffer_mask |= 1 << slot;
                            }

                            let mut dynamic_array_container_ty = var.ty;
                            if let naga::TypeInner::Struct { ref members, .. } =
                                module.types[var.ty].inner
                            {
                                dynamic_array_container_ty = members.last().unwrap().ty;
                            }
                            if let naga::TypeInner::Array {
                                size: naga::ArraySize::Dynamic,
                                ..
                            } = module.types[dynamic_array_container_ty].inner
                            {
                                sized_bindings.push(br);
                            }
                        }
                        _ => (),
                    }
                }
                let compiled_name = match info.entry_point_names[ep_index] {
                    Ok(ref name) => name.clone(),
                    Err(ref e) => {
                        return Err(super::ShaderCompilationError::Linkage(format!(
                            "MSL entry point error: {e}"
                        )))
                    }
                };

                entry_points.push(MslEntryPointCompileResult {
                    compiled_name,
                    workgroup_size: ep.workgroup_size,
                    wg_memory_sizes,
                    immutable_buffer_mask,
                    sized_bindings,
                });
            }

            Ok(MslCompileResult {
                shader,
                entry_points,
            })
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            unreachable!()
        }
    }
}
