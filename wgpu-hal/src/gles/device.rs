use super::conv;
use crate::util::map_naga_stage;
use glow::HasContext;
use std::{convert::TryInto, iter, ptr::NonNull, sync::Arc};

type ShaderStage<'a> = (
    naga::ShaderStage,
    &'a crate::ProgrammableStage<'a, super::Api>,
);
type NameBindingMap = fxhash::FxHashMap<String, (super::BindingRegister, u8)>;

struct CompilationContext<'a> {
    layout: &'a super::PipelineLayout,
    sampler_map: &'a mut super::SamplerBindMap,
    name_binding_map: &'a mut NameBindingMap,
}

impl CompilationContext<'_> {
    fn consume_reflection(
        self,
        module: &naga::Module,
        ep_info: &naga::valid::FunctionInfo,
        reflection_info: naga::back::glsl::ReflectionInfo,
    ) {
        for (handle, var) in module.global_variables.iter() {
            if ep_info[handle].is_empty() {
                continue;
            }
            let register = match var.class {
                naga::StorageClass::Uniform => super::BindingRegister::UniformBuffers,
                naga::StorageClass::Storage => super::BindingRegister::StorageBuffers,
                _ => continue,
            };

            let br = var.binding.as_ref().unwrap();
            let slot = self.layout.get_slot(br);

            let name = reflection_info.uniforms[&handle].clone();
            log::debug!(
                "Rebind buffer: {:?} -> {}, register={:?}, slot={}",
                var.name.as_ref(),
                &name,
                register,
                slot
            );
            self.name_binding_map.insert(name, (register, slot));
        }

        for (name, mapping) in reflection_info.texture_mapping {
            let var = &module.global_variables[mapping.texture];
            let register = if var.storage_access.is_empty() {
                super::BindingRegister::Textures
            } else {
                super::BindingRegister::Images
            };

            let tex_br = var.binding.as_ref().unwrap();
            let texture_linear_index = self.layout.get_slot(tex_br);

            self.name_binding_map
                .insert(name, (register, texture_linear_index));
            if let Some(sampler_handle) = mapping.sampler {
                let sam_br = module.global_variables[sampler_handle]
                    .binding
                    .as_ref()
                    .unwrap();
                let sampler_linear_index = self.layout.get_slot(sam_br);
                self.sampler_map[texture_linear_index as usize] = Some(sampler_linear_index);
            }
        }
    }
}

impl super::Device {
    unsafe fn compile_shader(
        &self,
        shader: &str,
        naga_stage: naga::ShaderStage,
    ) -> Result<glow::Shader, crate::PipelineError> {
        let gl = &self.shared.context;
        let target = match naga_stage {
            naga::ShaderStage::Vertex => glow::VERTEX_SHADER,
            naga::ShaderStage::Fragment => glow::FRAGMENT_SHADER,
            naga::ShaderStage::Compute => glow::COMPUTE_SHADER,
        };

        let raw = gl.create_shader(target).unwrap();
        gl.shader_source(raw, shader);
        gl.compile_shader(raw);

        log::info!("\tCompiled shader {:?}", raw);

        let compiled_ok = gl.get_shader_compile_status(raw);
        let msg = gl.get_shader_info_log(raw);
        if compiled_ok {
            if !msg.is_empty() {
                log::warn!("\tCompile: {}", msg);
            }
            Ok(raw)
        } else {
            Err(crate::PipelineError::Linkage(
                map_naga_stage(naga_stage),
                msg,
            ))
        }
    }

    fn create_shader(
        &self,
        naga_stage: naga::ShaderStage,
        stage: &crate::ProgrammableStage<super::Api>,
        context: CompilationContext,
    ) -> Result<glow::Shader, crate::PipelineError> {
        use naga::back::glsl;
        let options = glsl::Options {
            version: self.shared.shading_language_version,
            binding_map: Default::default(), //TODO
        };
        let pipeline_options = glsl::PipelineOptions {
            shader_stage: naga_stage,
            entry_point: stage.entry_point.to_string(),
        };

        let shader = &stage.module.naga;
        let entry_point_index = shader
            .module
            .entry_points
            .iter()
            .position(|ep| ep.name.as_str() == stage.entry_point)
            .ok_or(crate::PipelineError::EntryPoint(naga_stage))?;

        let mut output = String::new();
        let mut writer = glsl::Writer::new(
            &mut output,
            &shader.module,
            &shader.info,
            &options,
            &pipeline_options,
        )
        .map_err(|e| {
            let msg = format!("{}", e);
            crate::PipelineError::Linkage(map_naga_stage(naga_stage), msg)
        })?;

        let reflection_info = writer.write().map_err(|e| {
            let msg = format!("{}", e);
            crate::PipelineError::Linkage(map_naga_stage(naga_stage), msg)
        })?;

        log::debug!("Naga generated shader:\n{}", output);

        context.consume_reflection(
            &shader.module,
            shader.info.get_entry_point(entry_point_index),
            reflection_info,
        );

        unsafe { self.compile_shader(&output, naga_stage) }
    }

    unsafe fn create_pipeline<'a, I: Iterator<Item = ShaderStage<'a>>>(
        &self,
        shaders: I,
        layout: &super::PipelineLayout,
    ) -> Result<super::PipelineInner, crate::PipelineError> {
        let gl = &self.shared.context;
        let program = gl.create_program().unwrap();

        let mut name_binding_map = NameBindingMap::default();
        let mut sampler_map = [None; super::MAX_TEXTURE_SLOTS];
        let mut has_stages = wgt::ShaderStage::empty();
        let mut shaders_to_delete = arrayvec::ArrayVec::<[_; 3]>::new();

        for (naga_stage, stage) in shaders {
            has_stages |= map_naga_stage(naga_stage);
            let context = CompilationContext {
                layout,
                sampler_map: &mut sampler_map,
                name_binding_map: &mut name_binding_map,
            };

            let shader = self.create_shader(naga_stage, stage, context)?;
            shaders_to_delete.push(shader);
        }

        // Create empty fragment shader if only vertex shader is present
        if has_stages == wgt::ShaderStage::VERTEX {
            let version = match self.shared.shading_language_version {
                naga::back::glsl::Version::Embedded(v) => v,
                naga::back::glsl::Version::Desktop(_) => unreachable!(),
            };
            let shader_src = format!("#version {} es \n void main(void) {{}}", version,);
            log::info!("Only vertex shader is present. Creating an empty fragment shader",);
            let shader = self.compile_shader(&shader_src, naga::ShaderStage::Fragment)?;
            shaders_to_delete.push(shader);
        }

        for &shader in shaders_to_delete.iter() {
            gl.attach_shader(program, shader);
        }
        gl.link_program(program);

        for shader in shaders_to_delete {
            gl.delete_shader(shader);
        }

        log::info!("\tLinked program {:?}", program);

        let linked_ok = gl.get_program_link_status(program);
        let msg = gl.get_program_info_log(program);
        if !linked_ok {
            return Err(crate::PipelineError::Linkage(has_stages, msg));
        }
        if !msg.is_empty() {
            log::warn!("\tLink: {}", msg);
        }

        if !self
            .shared
            .private_caps
            .contains(super::PrivateCapability::SHADER_BINDING_LAYOUT)
        {
            // This remapping is only needed if we aren't able to put the binding layout
            // in the shader. We can't remap storage buffers this way.
            gl.use_program(Some(program));
            for (ref name, (register, slot)) in name_binding_map {
                log::trace!("Get binding {:?} from program {:?}", name, program);
                match register {
                    super::BindingRegister::UniformBuffers => {
                        let index = gl.get_uniform_block_index(program, name).unwrap();
                        gl.uniform_block_binding(program, index, slot as _);
                    }
                    super::BindingRegister::StorageBuffers => {
                        let index = gl.get_shader_storage_block_index(program, name).unwrap();
                        log::error!(
                            "Unable to re-map shader storage block {} to {}",
                            name,
                            index
                        );
                        return Err(crate::DeviceError::Lost.into());
                    }
                    super::BindingRegister::Textures | super::BindingRegister::Images => {
                        let loc = gl.get_uniform_location(program, name).unwrap();
                        gl.uniform_1_i32(Some(&loc), slot as _);
                    }
                }
            }
        }

        let uniforms = {
            let count = gl.get_active_uniforms(program);
            let mut offset = 0;
            let mut uniforms = Vec::new();

            for uniform in 0..count {
                let glow::ActiveUniform { size, utype, name } =
                    gl.get_active_uniform(program, uniform).unwrap();

                if let Some(location) = gl.get_uniform_location(program, &name) {
                    // Sampler2D won't show up in UniformLocation and the only other uniforms
                    // should be push constants
                    uniforms.push(super::UniformDesc {
                        location,
                        offset,
                        utype,
                    });

                    offset += size as u32;
                }
            }

            uniforms.into_boxed_slice()
        };

        Ok(super::PipelineInner {
            program,
            sampler_map,
            uniforms,
        })
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self) {
        let gl = &self.shared.context;
        gl.delete_vertex_array(self.main_vao);
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let gl = &self.shared.context;

        let target = if desc.usage.contains(crate::BufferUse::INDEX) {
            glow::ELEMENT_ARRAY_BUFFER
        } else {
            glow::ARRAY_BUFFER
        };

        let mut map_flags = 0;
        if desc
            .usage
            .intersects(crate::BufferUse::MAP_READ | crate::BufferUse::MAP_WRITE)
        {
            map_flags |= glow::MAP_PERSISTENT_BIT;
            if desc
                .memory_flags
                .contains(crate::MemoryFlag::PREFER_COHERENT)
            {
                map_flags |= glow::MAP_COHERENT_BIT;
            }
        }
        if desc.usage.contains(crate::BufferUse::MAP_READ) {
            map_flags |= glow::MAP_READ_BIT;
        }
        if desc.usage.contains(crate::BufferUse::MAP_WRITE) {
            map_flags |= glow::MAP_WRITE_BIT;
        }

        let raw = gl.create_buffer().unwrap();
        gl.bind_buffer(target, Some(raw));
        let raw_size = desc
            .size
            .try_into()
            .map_err(|_| crate::DeviceError::OutOfMemory)?;
        gl.buffer_storage(target, raw_size, None, map_flags);
        gl.bind_buffer(target, None);

        Ok(super::Buffer {
            raw,
            target,
            size: desc.size,
            map_flags,
        })
    }
    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        let gl = &self.shared.context;
        gl.delete_buffer(buffer.raw);
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let gl = &self.shared.context;

        let is_coherent = buffer.map_flags & glow::MAP_COHERENT_BIT != 0;
        let mut flags = buffer.map_flags | glow::MAP_UNSYNCHRONIZED_BIT;
        if !is_coherent {
            flags |= glow::MAP_FLUSH_EXPLICIT_BIT;
        }

        gl.bind_buffer(buffer.target, Some(buffer.raw));
        let ptr = gl.map_buffer_range(
            buffer.target,
            range.start as i32,
            (range.end - range.start) as i32,
            flags,
        );
        gl.bind_buffer(buffer.target, None);

        Ok(crate::BufferMapping {
            ptr: NonNull::new(ptr).ok_or(crate::DeviceError::Lost)?,
            is_coherent,
        })
    }
    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        let gl = &self.shared.context;
        gl.bind_buffer(buffer.target, Some(buffer.raw));
        gl.unmap_buffer(buffer.target);
        gl.bind_buffer(buffer.target, None);
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let gl = &self.shared.context;
        gl.bind_buffer(buffer.target, Some(buffer.raw));
        for range in ranges {
            gl.flush_mapped_buffer_range(
                buffer.target,
                range.start as i32,
                (range.end - range.start) as i32,
            );
        }
    }
    unsafe fn invalidate_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        let gl = &self.shared.context;
        gl.bind_buffer(buffer.target, Some(buffer.raw));
        for range in ranges {
            gl.invalidate_buffer_sub_data(
                buffer.target,
                range.start as i32,
                (range.end - range.start) as i32,
            );
        }
    }

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let gl = &self.shared.context;

        let render_usage = crate::TextureUse::COLOR_TARGET
            | crate::TextureUse::DEPTH_STENCIL_WRITE
            | crate::TextureUse::DEPTH_STENCIL_READ;
        let format_desc = self.shared.describe_texture_format(desc.format);

        let inner = if render_usage.contains(desc.usage)
            && desc.dimension == wgt::TextureDimension::D2
            && desc.size.depth_or_array_layers == 1
        {
            let raw = gl.create_renderbuffer().unwrap();
            gl.bind_renderbuffer(glow::RENDERBUFFER, Some(raw));
            if desc.sample_count > 1 {
                gl.renderbuffer_storage_multisample(
                    glow::RENDERBUFFER,
                    desc.sample_count as i32,
                    format_desc.internal,
                    desc.size.width as i32,
                    desc.size.height as i32,
                );
            } else {
                gl.renderbuffer_storage(
                    glow::RENDERBUFFER,
                    format_desc.internal,
                    desc.size.width as i32,
                    desc.size.height as i32,
                );
            }

            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            super::TextureInner::Renderbuffer { raw }
        } else {
            let raw = gl.create_texture().unwrap();
            let target = match desc.dimension {
                wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => {
                    if desc.sample_count > 1 {
                        let target = glow::TEXTURE_2D;
                        gl.bind_texture(target, Some(raw));
                        gl.tex_storage_2d_multisample(
                            target,
                            desc.sample_count as i32,
                            format_desc.internal,
                            desc.size.width as i32,
                            desc.size.height as i32,
                            true,
                        );
                        target
                    } else if desc.size.depth_or_array_layers > 1 {
                        let target = glow::TEXTURE_2D_ARRAY;
                        gl.bind_texture(target, Some(raw));
                        gl.tex_storage_3d(
                            target,
                            desc.mip_level_count as i32,
                            format_desc.internal,
                            desc.size.width as i32,
                            desc.size.height as i32,
                            desc.size.depth_or_array_layers as i32,
                        );
                        target
                    } else {
                        let target = glow::TEXTURE_2D;
                        gl.bind_texture(target, Some(raw));
                        gl.tex_storage_2d(
                            target,
                            desc.mip_level_count as i32,
                            format_desc.internal,
                            desc.size.width as i32,
                            desc.size.height as i32,
                        );
                        target
                    }
                }
                wgt::TextureDimension::D3 => {
                    let target = glow::TEXTURE_3D;
                    gl.bind_texture(target, Some(raw));
                    gl.tex_storage_3d(
                        target,
                        desc.mip_level_count as i32,
                        format_desc.internal,
                        desc.size.width as i32,
                        desc.size.height as i32,
                        desc.size.depth_or_array_layers as i32,
                    );
                    target
                }
            };

            match desc.format.describe().sample_type {
                wgt::TextureSampleType::Float { filterable: false }
                | wgt::TextureSampleType::Uint
                | wgt::TextureSampleType::Sint => {
                    // reset default filtering mode
                    gl.tex_parameter_i32(target, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                    gl.tex_parameter_i32(target, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                }
                wgt::TextureSampleType::Float { filterable: true }
                | wgt::TextureSampleType::Depth => {}
            }

            gl.bind_texture(target, None);
            super::TextureInner::Texture { raw, target }
        };

        Ok(super::Texture {
            inner,
            mip_level_count: desc.mip_level_count,
            array_layer_count: if desc.dimension == wgt::TextureDimension::D2 {
                desc.size.depth_or_array_layers
            } else {
                1
            },
            format: desc.format,
            format_desc,
        })
    }
    unsafe fn destroy_texture(&self, texture: super::Texture) {
        let gl = &self.shared.context;
        match texture.inner {
            super::TextureInner::Renderbuffer { raw, .. } => {
                gl.delete_renderbuffer(raw);
            }
            super::TextureInner::Texture { raw, .. } => {
                gl.delete_texture(raw);
            }
        }
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> Result<super::TextureView, crate::DeviceError> {
        let end_array_layer = match desc.range.array_layer_count {
            Some(count) => desc.range.base_array_layer + count.get(),
            None => texture.array_layer_count,
        };
        let end_mip_level = match desc.range.mip_level_count {
            Some(count) => desc.range.base_mip_level + count.get(),
            None => texture.mip_level_count,
        };
        Ok(super::TextureView {
            inner: match texture.inner {
                super::TextureInner::Renderbuffer { raw } => {
                    super::TextureInner::Renderbuffer { raw }
                }
                super::TextureInner::Texture { raw, target: _ } => super::TextureInner::Texture {
                    raw,
                    target: conv::map_view_dimension(desc.dimension),
                },
            },
            sample_type: texture.format.describe().sample_type,
            aspects: crate::FormatAspect::from(texture.format)
                & crate::FormatAspect::from(desc.range.aspect),
            mip_levels: desc.range.base_mip_level..end_mip_level,
            array_layers: desc.range.base_array_layer..end_array_layer,
        })
    }
    unsafe fn destroy_texture_view(&self, _view: super::TextureView) {}

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        let gl = &self.shared.context;

        let raw = gl.create_sampler().unwrap();

        let (min, mag) =
            conv::map_filter_modes(desc.min_filter, desc.mag_filter, desc.mipmap_filter);

        gl.sampler_parameter_i32(raw, glow::TEXTURE_MIN_FILTER, min as i32);
        gl.sampler_parameter_i32(raw, glow::TEXTURE_MAG_FILTER, mag as i32);

        gl.sampler_parameter_i32(
            raw,
            glow::TEXTURE_WRAP_S,
            conv::map_address_mode(desc.address_modes[0]) as i32,
        );
        gl.sampler_parameter_i32(
            raw,
            glow::TEXTURE_WRAP_T,
            conv::map_address_mode(desc.address_modes[1]) as i32,
        );
        gl.sampler_parameter_i32(
            raw,
            glow::TEXTURE_WRAP_R,
            conv::map_address_mode(desc.address_modes[2]) as i32,
        );

        if let Some(border_color) = desc.border_color {
            let border = match border_color {
                wgt::SamplerBorderColor::TransparentBlack => [0.0; 4],
                wgt::SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                wgt::SamplerBorderColor::OpaqueWhite => [1.0; 4],
            };
            gl.sampler_parameter_f32_slice(raw, glow::TEXTURE_BORDER_COLOR, &border);
        }

        if let Some(ref range) = desc.lod_clamp {
            gl.sampler_parameter_f32(raw, glow::TEXTURE_MIN_LOD, range.start);
            gl.sampler_parameter_f32(raw, glow::TEXTURE_MAX_LOD, range.end);
        }

        //TODO: `desc.anisotropy_clamp` depends on the downlevel flag
        // gl.sampler_parameter_f32(rawow::TEXTURE_MAX_ANISOTROPY, aniso as f32);

        //set_param_float(glow::TEXTURE_LOD_BIAS, info.lod_bias.0);

        if let Some(compare) = desc.compare {
            gl.sampler_parameter_i32(
                raw,
                glow::TEXTURE_COMPARE_MODE,
                glow::COMPARE_REF_TO_TEXTURE as i32,
            );
            gl.sampler_parameter_i32(
                raw,
                glow::TEXTURE_COMPARE_FUNC,
                conv::map_compare_func(compare) as i32,
            );
        }

        Ok(super::Sampler { raw })
    }
    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        let gl = &self.shared.context;
        gl.delete_sampler(sampler.raw);
    }

    unsafe fn create_command_encoder(
        &self,
        _desc: &crate::CommandEncoderDescriptor<super::Api>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        Ok(super::CommandEncoder {
            cmd_buffer: super::CommandBuffer::default(),
            state: Default::default(),
            private_caps: self.shared.private_caps,
        })
    }
    unsafe fn destroy_command_encoder(&self, _encoder: super::CommandEncoder) {}

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> Result<super::BindGroupLayout, crate::DeviceError> {
        Ok(super::BindGroupLayout {
            entries: Arc::from(desc.entries),
        })
    }
    unsafe fn destroy_bind_group_layout(&self, _bg_layout: super::BindGroupLayout) {}

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::Api>,
    ) -> Result<super::PipelineLayout, crate::DeviceError> {
        let mut group_infos = Vec::with_capacity(desc.bind_group_layouts.len());
        let mut num_samplers = 0u8;
        let mut num_textures = 0u8;
        let mut num_images = 0u8;
        let mut num_uniform_buffers = 0u8;
        let mut num_storage_buffers = 0u8;

        for bg_layout in desc.bind_group_layouts {
            // create a vector with the size enough to hold all the bindings, filled with `!0`
            let mut binding_to_slot = vec![
                !0;
                bg_layout
                    .entries
                    .last()
                    .map_or(0, |b| b.binding as usize + 1)
            ]
            .into_boxed_slice();

            for entry in bg_layout.entries.iter() {
                let counter = match entry.ty {
                    wgt::BindingType::Sampler { .. } => &mut num_samplers,
                    wgt::BindingType::Texture { .. } => &mut num_textures,
                    wgt::BindingType::StorageTexture { .. } => &mut num_images,
                    wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Uniform,
                        ..
                    } => &mut num_uniform_buffers,
                    wgt::BindingType::Buffer {
                        ty: wgt::BufferBindingType::Storage { .. },
                        ..
                    } => &mut num_storage_buffers,
                };

                binding_to_slot[entry.binding as usize] = *counter;
                *counter += entry.count.map_or(1, |c| c.get() as u8);
            }

            group_infos.push(super::BindGroupLayoutInfo {
                entries: Arc::clone(&bg_layout.entries),
                binding_to_slot,
            });
        }

        Ok(super::PipelineLayout {
            group_infos: group_infos.into_boxed_slice(),
        })
    }
    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: super::PipelineLayout) {}

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<super::Api>,
    ) -> Result<super::BindGroup, crate::DeviceError> {
        let mut contents = Vec::new();

        for (entry, layout) in desc.entries.iter().zip(desc.layout.entries.iter()) {
            let binding = match layout.ty {
                wgt::BindingType::Buffer { .. } => {
                    let bb = &desc.buffers[entry.resource_index as usize];
                    super::RawBinding::Buffer {
                        raw: bb.buffer.raw,
                        offset: bb.offset as i32,
                        size: match bb.size {
                            Some(s) => s.get() as i32,
                            None => (bb.buffer.size - bb.offset) as i32,
                        },
                    }
                }
                wgt::BindingType::Sampler { .. } => {
                    let sampler = desc.samplers[entry.resource_index as usize];
                    super::RawBinding::Sampler(sampler.raw)
                }
                wgt::BindingType::Texture { .. } => {
                    let view = desc.textures[entry.resource_index as usize].view;
                    match view.inner {
                        super::TextureInner::Renderbuffer { .. } => {
                            panic!("Unable to use a renderbuffer in a group")
                        }
                        super::TextureInner::Texture { raw, target } => {
                            super::RawBinding::Texture { raw, target }
                        }
                    }
                }
                wgt::BindingType::StorageTexture {
                    access,
                    format,
                    view_dimension,
                } => {
                    let view = desc.textures[entry.resource_index as usize].view;
                    let format_desc = self.shared.describe_texture_format(format);
                    match view.inner {
                        super::TextureInner::Renderbuffer { .. } => {
                            panic!("Unable to use a renderbuffer in a group")
                        }
                        super::TextureInner::Texture { raw, .. } => {
                            super::RawBinding::Image(super::ImageBinding {
                                raw,
                                mip_level: view.mip_levels.start,
                                array_layer: match view_dimension {
                                    wgt::TextureViewDimension::D2Array
                                    | wgt::TextureViewDimension::CubeArray => None,
                                    _ => Some(view.array_layers.start),
                                },
                                access: conv::map_storage_access(access),
                                format: format_desc.internal,
                            })
                        }
                    }
                }
            };
            contents.push(binding);
        }

        Ok(super::BindGroup {
            contents: contents.into_boxed_slice(),
        })
    }
    unsafe fn destroy_bind_group(&self, _group: super::BindGroup) {}

    unsafe fn create_shader_module(
        &self,
        _desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        Ok(super::ShaderModule {
            naga: match shader {
                crate::ShaderInput::SpirV(_) => {
                    panic!("`Features::SPIRV_SHADER_PASSTHROUGH` is not enabled")
                }
                crate::ShaderInput::Naga(naga) => naga,
            },
        })
    }
    unsafe fn destroy_shader_module(&self, _module: super::ShaderModule) {}

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        let shaders = iter::once((naga::ShaderStage::Vertex, &desc.vertex_stage)).chain(
            desc.fragment_stage
                .as_ref()
                .map(|fs| (naga::ShaderStage::Fragment, fs)),
        );
        let inner = self.create_pipeline(shaders, desc.layout)?;

        let (vertex_buffers, vertex_attributes) = {
            let mut buffers = Vec::new();
            let mut attributes = Vec::new();
            for (index, vb_layout) in desc.vertex_buffers.iter().enumerate() {
                buffers.push(super::VertexBufferDesc {
                    step: vb_layout.step_mode,
                    stride: vb_layout.array_stride as u32,
                });
                for vat in vb_layout.attributes.iter() {
                    let format_desc = conv::describe_vertex_format(vat.format);
                    attributes.push(super::AttributeDesc {
                        location: vat.shader_location,
                        offset: vat.offset as u32,
                        buffer_index: index as u32,
                        format_desc,
                    });
                }
            }
            (buffers.into_boxed_slice(), attributes.into_boxed_slice())
        };

        let color_targets = {
            let mut targets = Vec::new();
            for ct in desc.color_targets.iter() {
                targets.push(super::ColorTargetDesc {
                    mask: ct.write_mask,
                    blend: ct.blend.as_ref().map(conv::map_blend),
                });
            }
            //Note: if any of the states are different, and `INDEPENDENT_BLEND` flag
            // is not exposed, then this pipeline will not bind correctly.
            targets.into_boxed_slice()
        };

        Ok(super::RenderPipeline {
            inner,
            primitive: desc.primitive,
            vertex_buffers,
            vertex_attributes,
            color_targets,
            depth: desc.depth_stencil.as_ref().map(|ds| super::DepthState {
                function: conv::map_compare_func(ds.depth_compare),
                mask: ds.depth_write_enabled,
            }),
            depth_bias: desc
                .depth_stencil
                .as_ref()
                .map(|ds| ds.bias)
                .unwrap_or_default(),
            stencil: desc
                .depth_stencil
                .as_ref()
                .map(|ds| conv::map_stencil(&ds.stencil)),
        })
    }
    unsafe fn destroy_render_pipeline(&self, pipeline: super::RenderPipeline) {
        let gl = &self.shared.context;
        gl.delete_program(pipeline.inner.program);
    }

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        let shaders = iter::once((naga::ShaderStage::Compute, &desc.stage));
        let inner = self.create_pipeline(shaders, desc.layout)?;

        Ok(super::ComputePipeline { inner })
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: super::ComputePipeline) {
        let gl = &self.shared.context;
        gl.delete_program(pipeline.inner.program);
    }

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<super::QuerySet, crate::DeviceError> {
        let gl = &self.shared.context;

        let mut queries = Vec::with_capacity(desc.count as usize);
        for _ in 0..desc.count {
            let query = gl
                .create_query()
                .map_err(|_| crate::DeviceError::OutOfMemory)?;
            queries.push(query);
        }

        Ok(super::QuerySet {
            queries: queries.into_boxed_slice(),
            target: match desc.ty {
                wgt::QueryType::Occlusion => glow::ANY_SAMPLES_PASSED,
                _ => unimplemented!(),
            },
        })
    }
    unsafe fn destroy_query_set(&self, set: super::QuerySet) {
        let gl = &self.shared.context;
        for &query in set.queries.iter() {
            gl.delete_query(query);
        }
    }
    unsafe fn create_fence(&self) -> Result<super::Fence, crate::DeviceError> {
        Ok(super::Fence {
            last_completed: 0,
            pending: Vec::new(),
        })
    }
    unsafe fn destroy_fence(&self, fence: super::Fence) {
        let gl = &self.shared.context;
        for (_, sync) in fence.pending {
            gl.delete_sync(sync);
        }
    }
    unsafe fn get_fence_value(
        &self,
        fence: &super::Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        Ok(fence.get_latest(&self.shared.context))
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        wait_value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        if fence.last_completed < wait_value {
            let gl = &self.shared.context;
            let timeout_ns = (timeout_ms as u64 * 1_000_000).min(!0u32 as u64);
            let &(_, sync) = fence
                .pending
                .iter()
                .find(|&&(value, _)| value >= wait_value)
                .unwrap();
            match gl.client_wait_sync(sync, glow::SYNC_FLUSH_COMMANDS_BIT, timeout_ns as i32) {
                glow::TIMEOUT_EXPIRED => Ok(false),
                glow::CONDITION_SATISFIED | glow::ALREADY_SIGNALED => Ok(true),
                _ => Err(crate::DeviceError::Lost),
            }
        } else {
            Ok(true)
        }
    }

    unsafe fn start_capture(&self) -> bool {
        false
    }
    unsafe fn stop_capture(&self) {}
}
