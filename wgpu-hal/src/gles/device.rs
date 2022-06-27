use super::conv;
use crate::auxil::map_naga_stage;
use glow::HasContext;
use std::{
    convert::TryInto,
    iter, ptr,
    sync::{Arc, Mutex},
};

#[cfg(not(target_arch = "wasm32"))]
use std::mem;

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
            let register = match var.space {
                naga::AddressSpace::Uniform => super::BindingRegister::UniformBuffers,
                naga::AddressSpace::Storage { .. } => super::BindingRegister::StorageBuffers,
                _ => continue,
            };

            let br = var.binding.as_ref().unwrap();
            let slot = self.layout.get_slot(br);

            let name = match reflection_info.uniforms.get(&handle) {
                Some(name) => name.clone(),
                None => continue,
            };
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
            let register = match module.types[var.ty].inner {
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage { .. },
                    ..
                } => super::BindingRegister::Images,
                _ => super::BindingRegister::Textures,
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
        gl: &glow::Context,
        shader: &str,
        naga_stage: naga::ShaderStage,
        #[cfg_attr(target_arch = "wasm32", allow(unused))] label: Option<&str>,
    ) -> Result<glow::Shader, crate::PipelineError> {
        let target = match naga_stage {
            naga::ShaderStage::Vertex => glow::VERTEX_SHADER,
            naga::ShaderStage::Fragment => glow::FRAGMENT_SHADER,
            naga::ShaderStage::Compute => glow::COMPUTE_SHADER,
        };

        let raw = gl.create_shader(target).unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        if gl.supports_debug() {
            //TODO: remove all transmutes from `object_label`
            // https://github.com/grovesNL/glow/issues/186
            gl.object_label(glow::SHADER, mem::transmute(raw), label);
        }

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
        gl: &glow::Context,
        naga_stage: naga::ShaderStage,
        stage: &crate::ProgrammableStage<super::Api>,
        context: CompilationContext,
    ) -> Result<glow::Shader, crate::PipelineError> {
        use naga::back::glsl;
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

        use naga::proc::BoundsCheckPolicy;
        // The image bounds checks require the TEXTURE_LEVELS feature available in GL core 1.3+.
        let version = gl.version();
        let image_check = if !version.is_embedded && (version.major, version.minor) >= (1, 3) {
            BoundsCheckPolicy::ReadZeroSkipWrite
        } else {
            BoundsCheckPolicy::Unchecked
        };

        // Other bounds check are either provided by glsl or not implemented yet.
        let policies = naga::proc::BoundsCheckPolicies {
            index: BoundsCheckPolicy::Unchecked,
            buffer: BoundsCheckPolicy::Unchecked,
            image: image_check,
            binding_array: BoundsCheckPolicy::Unchecked,
        };

        let mut output = String::new();
        let mut writer = glsl::Writer::new(
            &mut output,
            &shader.module,
            &shader.info,
            &context.layout.naga_options,
            &pipeline_options,
            policies,
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

        unsafe { Self::compile_shader(gl, &output, naga_stage, stage.module.label.as_deref()) }
    }

    unsafe fn create_pipeline<'a, I: Iterator<Item = ShaderStage<'a>>>(
        &self,
        gl: &glow::Context,
        shaders: I,
        layout: &super::PipelineLayout,
        #[cfg_attr(target_arch = "wasm32", allow(unused))] label: Option<&str>,
    ) -> Result<super::PipelineInner, crate::PipelineError> {
        let program = gl.create_program().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(label) = label {
            if gl.supports_debug() {
                gl.object_label(glow::PROGRAM, mem::transmute(program), Some(label));
            }
        }

        let mut name_binding_map = NameBindingMap::default();
        let mut sampler_map = [None; super::MAX_TEXTURE_SLOTS];
        let mut has_stages = wgt::ShaderStages::empty();
        let mut shaders_to_delete = arrayvec::ArrayVec::<_, 3>::new();

        for (naga_stage, stage) in shaders {
            has_stages |= map_naga_stage(naga_stage);
            let context = CompilationContext {
                layout,
                sampler_map: &mut sampler_map,
                name_binding_map: &mut name_binding_map,
            };

            let shader = Self::create_shader(gl, naga_stage, stage, context)?;
            shaders_to_delete.push(shader);
        }

        // Create empty fragment shader if only vertex shader is present
        if has_stages == wgt::ShaderStages::VERTEX {
            let version = match self.shared.shading_language_version {
                naga::back::glsl::Version::Embedded(v) => v,
                naga::back::glsl::Version::Desktop(_) => unreachable!(),
            };
            let shader_src = format!("#version {} es \n void main(void) {{}}", version,);
            log::info!("Only vertex shader is present. Creating an empty fragment shader",);
            let shader = Self::compile_shader(
                gl,
                &shader_src,
                naga::ShaderStage::Fragment,
                Some("(wgpu internal) dummy fragment shader"),
            )?;
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
            .contains(super::PrivateCapabilities::SHADER_BINDING_LAYOUT)
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
                        gl.uniform_1_i32(
                            gl.get_uniform_location(program, name).as_ref(),
                            slot as _,
                        );
                    }
                }
            }
        }

        let mut uniforms: [super::UniformDesc; super::MAX_PUSH_CONSTANTS] = Default::default();
        let count = gl.get_active_uniforms(program);
        let mut offset = 0;

        for uniform in 0..count {
            let glow::ActiveUniform { utype, name, .. } =
                gl.get_active_uniform(program, uniform).unwrap();

            if conv::is_sampler(utype) {
                continue;
            }

            if let Some(location) = gl.get_uniform_location(program, &name) {
                if uniforms[offset / 4].location.is_some() {
                    panic!("Offset already occupied")
                }

                // `size` will always be 1 so we need to guess the real size from the type
                let uniform_size = conv::uniform_byte_size(utype);

                uniforms[offset / 4] = super::UniformDesc {
                    location: Some(location),
                    size: uniform_size,
                    utype,
                };

                offset += uniform_size as usize;
            }
        }

        Ok(super::PipelineInner {
            program,
            sampler_map,
            uniforms,
        })
    }
}

impl crate::Device<super::Api> for super::Device {
    unsafe fn exit(self, queue: super::Queue) {
        let gl = &self.shared.context.lock();
        gl.delete_vertex_array(self.main_vao);
        gl.delete_framebuffer(queue.draw_fbo);
        gl.delete_framebuffer(queue.copy_fbo);
        gl.delete_buffer(queue.zero_buffer);
    }

    unsafe fn create_buffer(
        &self,
        desc: &crate::BufferDescriptor,
    ) -> Result<super::Buffer, crate::DeviceError> {
        let target = if desc.usage.contains(crate::BufferUses::INDEX) {
            glow::ELEMENT_ARRAY_BUFFER
        } else {
            glow::ARRAY_BUFFER
        };

        let emulate_map = self
            .shared
            .workarounds
            .contains(super::Workarounds::EMULATE_BUFFER_MAP)
            || !self
                .shared
                .private_caps
                .contains(super::PrivateCapabilities::BUFFER_ALLOCATION);

        if emulate_map && desc.usage.intersects(crate::BufferUses::MAP_WRITE) {
            return Ok(super::Buffer {
                raw: None,
                target,
                size: desc.size,
                map_flags: 0,
                data: Some(Arc::new(Mutex::new(vec![0; desc.size as usize]))),
            });
        }

        let gl = &self.shared.context.lock();

        let target = if desc.usage.contains(crate::BufferUses::INDEX) {
            glow::ELEMENT_ARRAY_BUFFER
        } else {
            glow::ARRAY_BUFFER
        };

        let is_host_visible = desc
            .usage
            .intersects(crate::BufferUses::MAP_READ | crate::BufferUses::MAP_WRITE);
        let is_coherent = desc
            .memory_flags
            .contains(crate::MemoryFlags::PREFER_COHERENT);

        let mut map_flags = 0;
        if desc.usage.contains(crate::BufferUses::MAP_READ) {
            map_flags |= glow::MAP_READ_BIT;
        }
        if desc.usage.contains(crate::BufferUses::MAP_WRITE) {
            map_flags |= glow::MAP_WRITE_BIT;
        }

        let raw = Some(gl.create_buffer().unwrap());
        gl.bind_buffer(target, raw);
        let raw_size = desc
            .size
            .try_into()
            .map_err(|_| crate::DeviceError::OutOfMemory)?;

        if self
            .shared
            .private_caps
            .contains(super::PrivateCapabilities::BUFFER_ALLOCATION)
        {
            if is_host_visible {
                map_flags |= glow::MAP_PERSISTENT_BIT;
                if is_coherent {
                    map_flags |= glow::MAP_COHERENT_BIT;
                }
            }
            gl.buffer_storage(target, raw_size, None, map_flags);
        } else {
            assert!(!is_coherent);
            let usage = if is_host_visible {
                if desc.usage.contains(crate::BufferUses::MAP_READ) {
                    glow::STREAM_READ
                } else {
                    glow::DYNAMIC_DRAW
                }
            } else {
                glow::STATIC_DRAW
            };
            gl.buffer_data_size(target, raw_size, usage);
        }

        gl.bind_buffer(target, None);

        if !is_coherent && desc.usage.contains(crate::BufferUses::MAP_WRITE) {
            map_flags |= glow::MAP_FLUSH_EXPLICIT_BIT;
        }
        //TODO: do we need `glow::MAP_UNSYNCHRONIZED_BIT`?

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(label) = desc.label {
            if gl.supports_debug() {
                gl.object_label(glow::BUFFER, mem::transmute(raw), Some(label));
            }
        }

        let data = if emulate_map && desc.usage.contains(crate::BufferUses::MAP_READ) {
            Some(Arc::new(Mutex::new(vec![0; desc.size as usize])))
        } else {
            None
        };

        Ok(super::Buffer {
            raw,
            target,
            size: desc.size,
            map_flags,
            data,
        })
    }
    unsafe fn destroy_buffer(&self, buffer: super::Buffer) {
        if let Some(raw) = buffer.raw {
            let gl = &self.shared.context.lock();
            gl.delete_buffer(raw);
        }
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> Result<crate::BufferMapping, crate::DeviceError> {
        let is_coherent = buffer.map_flags & glow::MAP_COHERENT_BIT != 0;
        let ptr = match buffer.raw {
            None => {
                let mut vec = buffer.data.as_ref().unwrap().lock().unwrap();
                let slice = &mut vec.as_mut_slice()[range.start as usize..range.end as usize];
                slice.as_mut_ptr()
            }
            Some(raw) => {
                let gl = &self.shared.context.lock();
                gl.bind_buffer(buffer.target, Some(raw));
                let ptr = if let Some(ref map_read_allocation) = buffer.data {
                    let mut guard = map_read_allocation.lock().unwrap();
                    let slice = guard.as_mut_slice();
                    self.shared.get_buffer_sub_data(gl, buffer.target, 0, slice);
                    slice.as_mut_ptr()
                } else {
                    gl.map_buffer_range(
                        buffer.target,
                        range.start as i32,
                        (range.end - range.start) as i32,
                        buffer.map_flags,
                    )
                };
                gl.bind_buffer(buffer.target, None);
                ptr
            }
        };
        Ok(crate::BufferMapping {
            ptr: ptr::NonNull::new(ptr).ok_or(crate::DeviceError::Lost)?,
            is_coherent,
        })
    }
    unsafe fn unmap_buffer(&self, buffer: &super::Buffer) -> Result<(), crate::DeviceError> {
        if let Some(raw) = buffer.raw {
            if buffer.data.is_none() {
                let gl = &self.shared.context.lock();
                gl.bind_buffer(buffer.target, Some(raw));
                gl.unmap_buffer(buffer.target);
                gl.bind_buffer(buffer.target, None);
            }
        }
        Ok(())
    }
    unsafe fn flush_mapped_ranges<I>(&self, buffer: &super::Buffer, ranges: I)
    where
        I: Iterator<Item = crate::MemoryRange>,
    {
        if let Some(raw) = buffer.raw {
            let gl = &self.shared.context.lock();
            gl.bind_buffer(buffer.target, Some(raw));
            for range in ranges {
                gl.flush_mapped_buffer_range(
                    buffer.target,
                    range.start as i32,
                    (range.end - range.start) as i32,
                );
            }
        }
    }
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {
        //TODO: do we need to do anything?
    }

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> Result<super::Texture, crate::DeviceError> {
        let gl = &self.shared.context.lock();

        let render_usage = crate::TextureUses::COLOR_TARGET
            | crate::TextureUses::DEPTH_STENCIL_WRITE
            | crate::TextureUses::DEPTH_STENCIL_READ;
        let format_desc = self.shared.describe_texture_format(desc.format);

        let mut copy_size = crate::CopyExtent {
            width: desc.size.width,
            height: desc.size.height,
            depth: 1,
        };

        let (inner, is_cubemap) = if render_usage.contains(desc.usage)
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

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(label) = desc.label {
                if gl.supports_debug() {
                    gl.object_label(glow::RENDERBUFFER, mem::transmute(raw), Some(label));
                }
            }

            gl.bind_renderbuffer(glow::RENDERBUFFER, None);
            (super::TextureInner::Renderbuffer { raw }, false)
        } else {
            let raw = gl.create_texture().unwrap();
            let (target, is_3d, is_cubemap) = match desc.dimension {
                wgt::TextureDimension::D1 | wgt::TextureDimension::D2 => {
                    if desc.size.depth_or_array_layers > 1 {
                        //HACK: detect a cube map
                        let cube_count = if desc.size.width == desc.size.height
                            && desc.size.depth_or_array_layers % 6 == 0
                            && desc.sample_count == 1
                        {
                            Some(desc.size.depth_or_array_layers / 6)
                        } else {
                            None
                        };
                        match cube_count {
                            None => (glow::TEXTURE_2D_ARRAY, true, false),
                            Some(1) => (glow::TEXTURE_CUBE_MAP, false, true),
                            Some(_) => (glow::TEXTURE_CUBE_MAP_ARRAY, true, true),
                        }
                    } else {
                        (glow::TEXTURE_2D, false, false)
                    }
                }
                wgt::TextureDimension::D3 => {
                    copy_size.depth = desc.size.depth_or_array_layers;
                    (glow::TEXTURE_3D, true, false)
                }
            };

            gl.bind_texture(target, Some(raw));
            //Note: this has to be done before defining the storage!
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

            if is_3d {
                gl.tex_storage_3d(
                    target,
                    desc.mip_level_count as i32,
                    format_desc.internal,
                    desc.size.width as i32,
                    desc.size.height as i32,
                    desc.size.depth_or_array_layers as i32,
                );
            } else if desc.sample_count > 1 {
                gl.tex_storage_2d_multisample(
                    target,
                    desc.sample_count as i32,
                    format_desc.internal,
                    desc.size.width as i32,
                    desc.size.height as i32,
                    true,
                );
            } else {
                gl.tex_storage_2d(
                    target,
                    desc.mip_level_count as i32,
                    format_desc.internal,
                    desc.size.width as i32,
                    desc.size.height as i32,
                );
            }

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(label) = desc.label {
                if gl.supports_debug() {
                    gl.object_label(glow::TEXTURE, mem::transmute(raw), Some(label));
                }
            }

            gl.bind_texture(target, None);
            (super::TextureInner::Texture { raw, target }, is_cubemap)
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
            copy_size,
            is_cubemap,
        })
    }
    unsafe fn destroy_texture(&self, texture: super::Texture) {
        let gl = &self.shared.context.lock();
        match texture.inner {
            super::TextureInner::Renderbuffer { raw, .. } => {
                gl.delete_renderbuffer(raw);
            }
            super::TextureInner::DefaultRenderbuffer => {}
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
            //TODO: use `conv::map_view_dimension(desc.dimension)`?
            inner: texture.inner.clone(),
            sample_type: texture.format.describe().sample_type,
            aspects: crate::FormatAspects::from(texture.format)
                & crate::FormatAspects::from(desc.range.aspect),
            mip_levels: desc.range.base_mip_level..end_mip_level,
            array_layers: desc.range.base_array_layer..end_array_layer,
            format: texture.format,
        })
    }
    unsafe fn destroy_texture_view(&self, _view: super::TextureView) {}

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> Result<super::Sampler, crate::DeviceError> {
        let gl = &self.shared.context.lock();

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
                wgt::SamplerBorderColor::TransparentBlack | wgt::SamplerBorderColor::Zero => {
                    [0.0; 4]
                }
                wgt::SamplerBorderColor::OpaqueBlack => [0.0, 0.0, 0.0, 1.0],
                wgt::SamplerBorderColor::OpaqueWhite => [1.0; 4],
            };
            gl.sampler_parameter_f32_slice(raw, glow::TEXTURE_BORDER_COLOR, &border);
        }

        if let Some(ref range) = desc.lod_clamp {
            gl.sampler_parameter_f32(raw, glow::TEXTURE_MIN_LOD, range.start);
            gl.sampler_parameter_f32(raw, glow::TEXTURE_MAX_LOD, range.end);
        }

        if let Some(anisotropy) = desc.anisotropy_clamp {
            gl.sampler_parameter_i32(raw, glow::TEXTURE_MAX_ANISOTROPY, anisotropy.get() as i32);
        }

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

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(label) = desc.label {
            if gl.supports_debug() {
                gl.object_label(glow::SAMPLER, mem::transmute(raw), Some(label));
            }
        }

        Ok(super::Sampler { raw })
    }
    unsafe fn destroy_sampler(&self, sampler: super::Sampler) {
        let gl = &self.shared.context.lock();
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
        use naga::back::glsl;

        let mut group_infos = Vec::with_capacity(desc.bind_group_layouts.len());
        let mut num_samplers = 0u8;
        let mut num_textures = 0u8;
        let mut num_images = 0u8;
        let mut num_uniform_buffers = 0u8;
        let mut num_storage_buffers = 0u8;

        let mut writer_flags = glsl::WriterFlags::ADJUST_COORDINATE_SPACE;
        writer_flags.set(
            glsl::WriterFlags::TEXTURE_SHADOW_LOD,
            self.shared
                .private_caps
                .contains(super::PrivateCapabilities::SHADER_TEXTURE_SHADOW_LOD),
        );
        let mut binding_map = glsl::BindingMap::default();

        for (group_index, bg_layout) in desc.bind_group_layouts.iter().enumerate() {
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
                let br = naga::ResourceBinding {
                    group: group_index as u32,
                    binding: entry.binding,
                };
                binding_map.insert(br, *counter);
                *counter += entry.count.map_or(1, |c| c.get() as u8);
            }

            group_infos.push(super::BindGroupLayoutInfo {
                entries: Arc::clone(&bg_layout.entries),
                binding_to_slot,
            });
        }

        Ok(super::PipelineLayout {
            group_infos: group_infos.into_boxed_slice(),
            naga_options: glsl::Options {
                version: self.shared.shading_language_version,
                writer_flags,
                binding_map,
            },
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
                        raw: bb.buffer.raw.unwrap(),
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
                    if view.mip_levels.start != 0 || view.array_layers.start != 0 {
                        log::error!("Unable to create a sampled texture binding for non-zero mipmap level or array layer.\n{}",
                            "This is an implementation problem of wgpu-hal/gles backend.")
                    }
                    let (raw, target) = view.inner.as_native();
                    super::RawBinding::Texture { raw, target }
                }
                wgt::BindingType::StorageTexture {
                    access,
                    format,
                    view_dimension,
                } => {
                    let view = desc.textures[entry.resource_index as usize].view;
                    let format_desc = self.shared.describe_texture_format(format);
                    let (raw, _target) = view.inner.as_native();
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
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        Ok(super::ShaderModule {
            naga: match shader {
                crate::ShaderInput::SpirV(_) => {
                    panic!("`Features::SPIRV_SHADER_PASSTHROUGH` is not enabled")
                }
                crate::ShaderInput::Naga(naga) => naga,
            },
            label: desc.label.map(|str| str.to_string()),
        })
    }
    unsafe fn destroy_shader_module(&self, _module: super::ShaderModule) {}

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<super::Api>,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        let gl = &self.shared.context.lock();
        let shaders = iter::once((naga::ShaderStage::Vertex, &desc.vertex_stage)).chain(
            desc.fragment_stage
                .as_ref()
                .map(|fs| (naga::ShaderStage::Fragment, fs)),
        );
        let inner = self.create_pipeline(gl, shaders, desc.layout, desc.label)?;

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
            for ct in desc.color_targets.iter().filter_map(|at| at.as_ref()) {
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
        let gl = &self.shared.context.lock();
        gl.delete_program(pipeline.inner.program);
    }

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<super::Api>,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        let gl = &self.shared.context.lock();
        let shaders = iter::once((naga::ShaderStage::Compute, &desc.stage));
        let inner = self.create_pipeline(gl, shaders, desc.layout, desc.label)?;

        Ok(super::ComputePipeline { inner })
    }
    unsafe fn destroy_compute_pipeline(&self, pipeline: super::ComputePipeline) {
        let gl = &self.shared.context.lock();
        gl.delete_program(pipeline.inner.program);
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> Result<super::QuerySet, crate::DeviceError> {
        let gl = &self.shared.context.lock();
        let mut temp_string = String::new();

        let mut queries = Vec::with_capacity(desc.count as usize);
        for i in 0..desc.count {
            let query = gl
                .create_query()
                .map_err(|_| crate::DeviceError::OutOfMemory)?;
            #[cfg(not(target_arch = "wasm32"))]
            if gl.supports_debug() {
                use std::fmt::Write;

                if let Some(label) = desc.label {
                    temp_string.clear();
                    let _ = write!(temp_string, "{}[{}]", label, i);
                    gl.object_label(glow::QUERY, mem::transmute(query), Some(&temp_string));
                }
            }
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
        let gl = &self.shared.context.lock();
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
        let gl = &self.shared.context.lock();
        for (_, sync) in fence.pending {
            gl.delete_sync(sync);
        }
    }
    unsafe fn get_fence_value(
        &self,
        fence: &super::Fence,
    ) -> Result<crate::FenceValue, crate::DeviceError> {
        #[cfg_attr(target_arch = "wasm32", allow(clippy::needless_borrow))]
        Ok(fence.get_latest(&self.shared.context.lock()))
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        wait_value: crate::FenceValue,
        timeout_ms: u32,
    ) -> Result<bool, crate::DeviceError> {
        if fence.last_completed < wait_value {
            let gl = &self.shared.context.lock();
            let timeout_ns = if cfg!(target_arch = "wasm32") {
                0
            } else {
                (timeout_ms as u64 * 1_000_000).min(!0u32 as u64)
            };
            let &(_, sync) = fence
                .pending
                .iter()
                .find(|&&(value, _)| value >= wait_value)
                .unwrap();
            match gl.client_wait_sync(sync, glow::SYNC_FLUSH_COMMANDS_BIT, timeout_ns as i32) {
                // for some reason firefox returns WAIT_FAILED, to investigate
                #[cfg(target_arch = "wasm32")]
                glow::WAIT_FAILED => {
                    log::warn!("wait failed!");
                    Ok(false)
                }
                glow::TIMEOUT_EXPIRED => Ok(false),
                glow::CONDITION_SATISFIED | glow::ALREADY_SIGNALED => Ok(true),
                _ => Err(crate::DeviceError::Lost),
            }
        } else {
            Ok(true)
        }
    }

    unsafe fn start_capture(&self) -> bool {
        #[cfg(feature = "renderdoc")]
        return self
            .render_doc
            .start_frame_capture(self.shared.context.raw_context(), ptr::null_mut());
        #[allow(unreachable_code)]
        false
    }
    unsafe fn stop_capture(&self) {
        #[cfg(feature = "renderdoc")]
        self.render_doc
            .end_frame_capture(ptr::null_mut(), ptr::null_mut())
    }
}

// SAFE: WASM doesn't have threads
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for super::Device {}
#[cfg(target_arch = "wasm32")]
unsafe impl Send for super::Device {}
