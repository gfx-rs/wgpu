use core::num::NonZeroU32;

use alloc::borrow::Cow;
#[cfg(feature = "naga-dep")]
use naga::back::glsl::{self, Options};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GlslCompileOptionsDesc {
    pub api_version: wst::glsl::ApiVersion,
    pub glsl_version: wst::glsl::GlslVersion,
    pub shader_texture_shadow_lod: bool,
    pub shader_draw_parameters: bool,
    pub binding_map: wst::glsl::BindingMap,
}
impl GlslCompileOptionsDesc {
    pub fn maximum_compat() -> Self {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct GlslCompileOptions {
    #[cfg(feature = "naga-dep")]
    pub(crate) options: Options,
    #[cfg(feature = "naga-dep")]
    pub api_version: wst::glsl::ApiVersion,
    #[cfg(feature = "naga-dep")]
    pub glsl_version: wst::glsl::GlslVersion,
    #[cfg(feature = "naga-dep")]
    pub binding_map: wst::glsl::BindingMap,
}
impl GlslCompileOptions {
    pub fn new(desc: GlslCompileOptionsDesc) -> Self {
        #[cfg(feature = "naga-dep")]
        {
            let mut writer_flags = glsl::WriterFlags::ADJUST_COORDINATE_SPACE;
            writer_flags.set(
                glsl::WriterFlags::TEXTURE_SHADOW_LOD,
                desc.shader_texture_shadow_lod,
            );
            writer_flags.set(
                glsl::WriterFlags::DRAW_PARAMETERS,
                desc.shader_draw_parameters,
            );
            // We always force point size to be written and it will be ignored by the driver if it's not a point list primitive.
            // https://github.com/gfx-rs/wgpu/pull/3440/files#r1095726950
            writer_flags.set(glsl::WriterFlags::FORCE_POINT_SIZE, true);
            Self {
                options: Options {
                    version: desc.glsl_version,
                    writer_flags: naga::back::glsl::WriterFlags::empty(),
                    binding_map: Default::default(),
                    zero_initialize_workgroup_memory: true,
                },
                api_version: desc.api_version,
                glsl_version: desc.glsl_version,
                binding_map: desc.binding_map,
            }
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            Self {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct GlslShaderDesc<'a> {
    pub options: Cow<'a, GlslCompileOptions>,
    pub stage: wst::ShaderStage,
    pub entry_point: &'a str,
    pub multiview_mask: Option<NonZeroU32>,
    pub zero_init_memory: bool,
    pub constants: &'a wst::PipelineConstants,
}

impl super::NagaShader {
    pub fn compile_glsl(
        &self,
        desc: GlslShaderDesc,
    ) -> Result<(String, wst::glsl::GlslReflectionInfo), crate::ShaderCompilationError> {
        #[cfg(feature = "naga-dep")]
        {
            let pipeline_options = glsl::PipelineOptions {
                shader_stage: desc.stage,
                entry_point: desc.entry_point.to_owned(),
                multiview: desc
                    .multiview_mask
                    .map(|a| NonZeroU32::new(a.get().count_ones()).unwrap()),
            };

            let (module, info) = naga::back::pipeline_constants::process_overrides(
                &self.module,
                &self.info,
                Some((desc.stage, desc.entry_point)),
                desc.constants,
            )
            .map_err(|e| {
                let msg = format!("{e}");
                crate::ShaderCompilationError::PipelineConstants(msg)
            })?;

            let entry_point_index = module
                .entry_points
                .iter()
                .position(|ep| ep.name.as_str() == desc.entry_point)
                .ok_or(crate::ShaderCompilationError::EntryPoint)?;

            use naga::proc::BoundsCheckPolicy;
            // The image bounds checks require the TEXTURE_LEVELS feature available in GL core 4.3+.
            let image_check = if !desc.options.api_version.is_embedded
                && desc.options.api_version.version >= 430
            {
                BoundsCheckPolicy::ReadZeroSkipWrite
            } else {
                BoundsCheckPolicy::Unchecked
            };

            // Other bounds check are either provided by glsl or not implemented yet.
            let policies = naga::proc::BoundsCheckPolicies {
                index: BoundsCheckPolicy::Unchecked,
                buffer: BoundsCheckPolicy::Unchecked,
                image_load: image_check,
                binding_array: BoundsCheckPolicy::Unchecked,
            };

            let mut output = String::new();
            let needs_temp_options =
                desc.zero_init_memory != desc.options.options.zero_initialize_workgroup_memory;
            let mut temp_options;
            let naga_options = if needs_temp_options {
                // We use a conditional here, as cloning the naga_options could be expensive
                // That is, we want to avoid doing that unless we cannot avoid it
                temp_options = desc.options.options.clone();
                temp_options.zero_initialize_workgroup_memory = desc.zero_init_memory;
                &temp_options
            } else {
                &desc.options.options
            };
            let mut writer = glsl::Writer::new(
                &mut output,
                &module,
                &info,
                naga_options,
                &pipeline_options,
                policies,
            )
            .map_err(|e| {
                let msg = format!("{e}");
                crate::ShaderCompilationError::Linkage(msg)
            })?;

            let reflection_info = writer.write().map_err(|e| {
                let msg = format!("{e}");
                crate::ShaderCompilationError::Linkage(msg)
            })?;

            let mut name_binding_map = Default::default();

            let entry_point_index = self
                .module
                .entry_points
                .iter()
                .position(|ep| ep.name.as_str() == desc.entry_point)
                .ok_or(crate::ShaderCompilationError::EntryPoint)?;
            let ep_info = self.info.get_entry_point(entry_point_index);

            for (handle, var) in module.global_variables.iter() {
                if ep_info[handle].is_empty() {
                    continue;
                }
                let register = match var.space {
                    naga::AddressSpace::Uniform => wst::glsl::BindingRegister::UniformBuffers,
                    naga::AddressSpace::Storage { .. } => {
                        wst::glsl::BindingRegister::StorageBuffers
                    }
                    _ => continue,
                };

                let br = var.binding.as_ref().unwrap();
                let slot = self.layout.get_slot(br);

                let name = match reflection_info.uniforms.get(&handle) {
                    Some(name) => name.clone(),
                    None => continue,
                };
                name_binding_map.insert(name, (register, slot));
            }

            for (name, mapping) in reflection_info.texture_mapping {
                let var = &module.global_variables[mapping.texture];
                let register = match module.types[var.ty].inner {
                    naga::TypeInner::Image {
                        class: naga::ImageClass::Storage { .. },
                        ..
                    } => wst::glsl::BindingRegister::Images,
                    _ => wst::glsl::BindingRegister::Textures,
                };

                let tex_br = var.binding.as_ref().unwrap();
                let texture_linear_index = self.layout.get_slot(tex_br);

                name_binding_map.insert(name, (register, texture_linear_index));
                if let Some(sampler_handle) = mapping.sampler {
                    let sam_br = module.global_variables[sampler_handle]
                        .binding
                        .as_ref()
                        .unwrap();
                    let sampler_linear_index = self.layout.get_slot(sam_br);
                    self.sampler_map[texture_linear_index as usize] = Some(sampler_linear_index);
                }
            }
            let immediates_items = reflection_info
                .immediates_items
                .into_iter()
                .map(|e| {
                    let ty_inner = &self.module.types[e.ty].inner;
                    wst::glsl::GlslImmediateItem {
                        access_path: e.access_path,
                        ty: ty_inner.try_into().unwrap(),
                        offset: e.offset,
                        size: ty_inner.size(self.module.to_ctx()),
                    }
                })
                .collect();
            let out_reflect = wst::glsl::GlslReflectionInfo {
                varying: reflection_info.varying,
                immediates_items,
                clip_distance_count: reflection_info.clip_distance_count,
                name_binding_map,
            };
            Ok((output, out_reflect))
        }
        #[cfg(not(feature = "naga-dep"))]
        {
            unreachable!()
        }
    }
}
