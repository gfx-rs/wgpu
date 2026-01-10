//! This is a player library for WebGPU traces.

#![cfg(not(target_arch = "wasm32"))]
#![warn(clippy::allow_attributes, unsafe_op_in_unsafe_fn)]

extern crate wgpu_core as wgc;
extern crate wgpu_types as wgt;

use std::{borrow::Cow, convert::Infallible, sync::Arc};

use hashbrown::HashMap;

use wgc::{
    binding_model::BindingResource,
    command::{ArcCommand, ArcReferences, BasePass, Command, PointerReferences},
    device::trace::{self, DataKind, DataLoader},
    id::PointerId,
};

pub struct Player {
    pipeline_layouts: HashMap<
        wgc::id::PointerId<wgc::id::markers::PipelineLayout>,
        Arc<wgc::binding_model::PipelineLayout>,
    >,
    shader_modules: HashMap<
        wgc::id::PointerId<wgc::id::markers::ShaderModule>,
        Arc<wgc::pipeline::ShaderModule>,
    >,
    bind_group_layouts: HashMap<
        wgc::id::PointerId<wgc::id::markers::BindGroupLayout>,
        Arc<wgc::binding_model::BindGroupLayout>,
    >,
    bind_groups: HashMap<
        wgc::id::PointerId<wgc::id::markers::BindGroup>,
        Arc<wgc::binding_model::BindGroup>,
    >,
    render_bundles: HashMap<
        wgc::id::PointerId<wgc::id::markers::RenderBundle>,
        Arc<wgc::command::RenderBundle>,
    >,
    render_pipelines: HashMap<
        wgc::id::PointerId<wgc::id::markers::RenderPipeline>,
        Arc<wgc::pipeline::RenderPipeline>,
    >,
    compute_pipelines: HashMap<
        wgc::id::PointerId<wgc::id::markers::ComputePipeline>,
        Arc<wgc::pipeline::ComputePipeline>,
    >,
    pipeline_caches: HashMap<
        wgc::id::PointerId<wgc::id::markers::PipelineCache>,
        Arc<wgc::pipeline::PipelineCache>,
    >,
    query_sets:
        HashMap<wgc::id::PointerId<wgc::id::markers::QuerySet>, Arc<wgc::resource::QuerySet>>,
    buffers: HashMap<wgc::id::PointerId<wgc::id::markers::Buffer>, Arc<wgc::resource::Buffer>>,
    textures: HashMap<wgc::id::PointerId<wgc::id::markers::Texture>, Arc<wgc::resource::Texture>>,
    texture_views:
        HashMap<wgc::id::PointerId<wgc::id::markers::TextureView>, Arc<wgc::resource::TextureView>>,
    external_textures: HashMap<
        wgc::id::PointerId<wgc::id::markers::ExternalTexture>,
        Arc<wgc::resource::ExternalTexture>,
    >,
    samplers: HashMap<wgc::id::PointerId<wgc::id::markers::Sampler>, Arc<wgc::resource::Sampler>>,
    blas_s: HashMap<wgc::id::PointerId<wgc::id::markers::Blas>, Arc<wgc::resource::Blas>>,
    tlas_s: HashMap<wgc::id::PointerId<wgc::id::markers::Tlas>, Arc<wgc::resource::Tlas>>,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            pipeline_layouts: HashMap::new(),
            shader_modules: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            bind_groups: HashMap::new(),
            render_bundles: HashMap::new(),
            render_pipelines: HashMap::new(),
            compute_pipelines: HashMap::new(),
            pipeline_caches: HashMap::new(),
            query_sets: HashMap::new(),
            buffers: HashMap::new(),
            textures: HashMap::new(),
            texture_views: HashMap::new(),
            external_textures: HashMap::new(),
            samplers: HashMap::new(),
            blas_s: HashMap::new(),
            tlas_s: HashMap::new(),
        }
    }
}

impl Player {
    pub fn process(
        &mut self,
        device: &Arc<wgc::device::Device>,
        queue: &Arc<wgc::device::queue::Queue>,
        action: trace::Action<PointerReferences>,
        loader: impl DataLoader,
    ) {
        use wgc::device::trace::Action;
        log::debug!("action {action:?}");
        match action {
            Action::Init { .. } => {
                panic!("Unexpected Action::Init: has to be the first action only")
            }
            Action::ConfigureSurface { .. }
            | Action::Present(_)
            | Action::DiscardSurfaceTexture(_) => {
                panic!("Unexpected Surface action: winit feature is not enabled")
            }
            Action::CreateBuffer(id, desc) => {
                let buffer = device.create_buffer(&desc).expect("create_buffer error");
                self.buffers.insert(id, buffer);
            }
            Action::FreeBuffer(id) => {
                // Note: buffer remains in the HashMap. "Free" and "Destroy"
                // mean the opposite from WebGPU.
                let buffer = self.buffers.get(&id).expect("invalid buffer");
                buffer.destroy();
            }
            Action::DestroyBuffer(id) => {
                let buffer = self.buffers.remove(&id).expect("invalid buffer");
                let _ = buffer.unmap();
            }
            Action::CreateTexture(id, desc) => {
                let texture = device.create_texture(&desc).expect("create_texture error");
                self.textures.insert(id, texture);
            }
            Action::FreeTexture(id) => {
                // Note: texture remains in the HashMap. "Free" and "Destroy"
                // mean the opposite from WebGPU.
                let texture = self.textures.get(&id).expect("invalid texture");
                texture.destroy();
            }
            Action::DestroyTexture(id) => {
                self.textures.remove(&id).expect("invalid texture");
            }
            Action::CreateTextureView { id, parent, desc } => {
                let parent_texture = self.resolve_texture_id(parent);
                let texture_view = device
                    .create_texture_view(&parent_texture, &desc)
                    .expect("create_texture_view error");
                self.texture_views.insert(id, texture_view);
            }
            Action::DestroyTextureView(id) => {
                self.texture_views
                    .remove(&id)
                    .expect("invalid texture view");
            }
            Action::CreateExternalTexture { id, desc, planes } => {
                let planes = planes
                    .iter()
                    .map(|&id| self.resolve_texture_view_id(id))
                    .collect::<Vec<_>>();
                let external_texture = device
                    .create_external_texture(&desc, &planes)
                    .expect("create_external_texture error");
                self.external_textures.insert(id, external_texture);
            }
            Action::FreeExternalTexture(id) => {
                // Note: external texture remains in the HashMap. "Free" and "Destroy"
                // mean the opposite from WebGPU.
                let external_texture = self
                    .external_textures
                    .get(&id)
                    .expect("invalid external texture");
                external_texture.destroy();
            }
            Action::DestroyExternalTexture(id) => {
                self.external_textures
                    .remove(&id)
                    .expect("invalid external texture");
            }
            Action::CreateSampler(id, desc) => {
                let sampler = device.create_sampler(&desc).expect("create_sampler error");
                self.samplers.insert(id, sampler);
            }
            Action::DestroySampler(id) => {
                self.samplers.remove(&id).expect("invalid sampler");
            }
            Action::GetSurfaceTexture { .. } => {
                unimplemented!()
            }
            Action::CreateBindGroupLayout(id, desc) => {
                let bind_group_layout = device
                    .create_bind_group_layout(&desc)
                    .expect("create_bind_group_layout error");
                self.bind_group_layouts.insert(id, bind_group_layout);
            }
            Action::DestroyBindGroupLayout(id) => {
                self.bind_group_layouts
                    .remove(&id)
                    .expect("invalid bind group layout");
            }
            Action::CreatePipelineLayout(id, desc) => {
                let bind_group_layouts: Vec<Arc<wgc::binding_model::BindGroupLayout>> = desc
                    .bind_group_layouts
                    .to_vec()
                    .into_iter()
                    .map(|bgl_id| self.resolve_bind_group_layout_id(bgl_id))
                    .collect();

                let resolved_desc = wgc::binding_model::ResolvedPipelineLayoutDescriptor {
                    label: desc.label.clone(),
                    bind_group_layouts: Cow::from(&bind_group_layouts),
                    immediate_size: desc.immediate_size,
                };

                let pipeline_layout = device
                    .create_pipeline_layout(&resolved_desc)
                    .expect("create_pipeline_layout error");
                self.pipeline_layouts.insert(id, pipeline_layout);
            }
            Action::DestroyPipelineLayout(id) => {
                self.pipeline_layouts
                    .remove(&id)
                    .expect("invalid pipeline layout");
            }
            Action::CreateBindGroup(id, desc) => {
                let resolved_desc = self.resolve_bind_group_descriptor(desc);
                let bind_group = device
                    .create_bind_group(resolved_desc)
                    .expect("create_bind_group error");
                self.bind_groups.insert(id, bind_group);
            }
            Action::DestroyBindGroup(id) => {
                let _bind_group = self.bind_groups.remove(&id).expect("invalid bind group");
            }
            Action::CreateShaderModule { id, desc, data } => {
                let code = loader.load_utf8(&data);
                let source = if data.kind() == DataKind::Wgsl {
                    wgc::pipeline::ShaderModuleSource::Wgsl(code.clone())
                } else if data.kind() == DataKind::Ron {
                    let module = ron::de::from_str(&code).unwrap();
                    wgc::pipeline::ShaderModuleSource::Naga(module)
                } else {
                    panic!(
                        "Unknown data kind for CreateShaderModule: {:?}",
                        data.kind()
                    );
                };
                match device.create_shader_module(&desc, source) {
                    Ok(module) => self.shader_modules.insert(id, module),
                    Err(e) => panic!("shader compilation error:\n---{code}\n---\n{e}"),
                };
            }
            Action::CreateShaderModulePassthrough {
                id,
                data,
                entry_point,
                label,
                num_workgroups,
                runtime_checks,
            } => {
                let spirv = data.iter().find_map(|a| {
                    if a.kind() == DataKind::Spv {
                        let data = loader.load(a);
                        assert!(data.len().is_multiple_of(4));

                        Some(Cow::Owned(bytemuck::pod_collect_to_vec(&data)))
                    } else {
                        None
                    }
                });
                let dxil = data
                    .iter()
                    .find_map(|a| (a.kind() == DataKind::Dxil).then(|| loader.load(a)));
                let hlsl = data
                    .iter()
                    .find_map(|a| (a.kind() == DataKind::Hlsl).then(|| loader.load_utf8(a)));
                let msl = data
                    .iter()
                    .find_map(|a| (a.kind() == DataKind::Msl).then(|| loader.load_utf8(a)));
                let glsl = data
                    .iter()
                    .find_map(|a| (a.kind() == DataKind::Glsl).then(|| loader.load_utf8(a)));
                let wgsl = data
                    .iter()
                    .find_map(|a| (a.kind() == DataKind::Wgsl).then(|| loader.load_utf8(a)));
                let desc = wgt::CreateShaderModuleDescriptorPassthrough {
                    entry_point,
                    label,
                    num_workgroups,
                    runtime_checks,

                    spirv,
                    dxil,
                    hlsl,
                    msl,
                    glsl,
                    wgsl,
                };
                match unsafe { device.create_shader_module_passthrough(&desc) } {
                    Ok(module) => self.shader_modules.insert(id, module),
                    Err(e) => panic!("shader compilation error:\n{e}"),
                };
            }
            Action::DestroyShaderModule(id) => {
                self.shader_modules
                    .remove(&id)
                    .expect("invalid shader module");
            }
            Action::CreateComputePipeline { id, desc } => {
                let resolved_desc = self.resolve_compute_pipeline_descriptor(desc);
                let pipeline = device
                    .create_compute_pipeline(resolved_desc)
                    .expect("create_compute_pipeline error");
                self.compute_pipelines.insert(id, pipeline);
            }
            Action::DestroyComputePipeline(id) => {
                self.compute_pipelines
                    .remove(&id)
                    .expect("invalid compute pipeline");
            }
            Action::CreateGeneralRenderPipeline { id, desc } => {
                // Note that this is the `General` version of the render
                // pipeline descriptor that can represent either a conventional
                // pipeline or a mesh shading pipeline.
                let resolved_desc = self.resolve_render_pipeline_descriptor(desc);
                let pipeline = device
                    .create_render_pipeline(resolved_desc)
                    .expect("create_render_pipeline error");
                self.render_pipelines.insert(id, pipeline);
            }
            Action::DestroyRenderPipeline(id) => {
                self.render_pipelines
                    .remove(&id)
                    .expect("invalid render pipeline");
            }
            Action::CreatePipelineCache { id, desc } => {
                let cache = unsafe { device.create_pipeline_cache(&desc) }.unwrap();
                self.pipeline_caches.insert(id, cache);
            }
            Action::DestroyPipelineCache(id) => {
                self.pipeline_caches
                    .remove(&id)
                    .expect("invalid pipeline cache");
            }
            Action::CreateRenderBundle { .. } => {
                unimplemented!("traced render bundles are not supported");
            }
            Action::DestroyRenderBundle(id) => {
                self.render_bundles
                    .remove(&id)
                    .expect("invalid render bundle");
            }
            Action::CreateQuerySet { id, desc } => {
                let query_set = device
                    .create_query_set(&desc)
                    .expect("create_query_set error");
                self.query_sets.insert(id, query_set);
            }
            Action::DestroyQuerySet(id) => {
                self.query_sets.remove(&id).expect("invalid query set");
            }
            Action::WriteBuffer {
                id,
                data,
                range,
                queued,
            } => {
                let buffer = self.resolve_buffer_id(id);
                let bin = loader.load(&data);
                let size = (range.end - range.start) as usize;
                if queued {
                    queue
                        .write_buffer(buffer, range.start, &bin)
                        .expect("Queue::write_buffer error");
                } else {
                    device
                        .set_buffer_data(&buffer, range.start, &bin[..size])
                        .expect("Device::set_buffer_data error");
                }
            }
            Action::WriteTexture {
                to,
                data,
                layout,
                size,
            } => {
                let to = self.resolve_texel_copy_texture_info(to);
                let bin = loader.load(&data);
                queue
                    .write_texture(to, &bin, &layout, &size)
                    .expect("Queue::write_texture error");
            }
            Action::Submit(_index, ref commands) if commands.is_empty() => {
                queue.submit(&[]).unwrap();
            }
            Action::Submit(_index, commands) => {
                let resolved_commands: Vec<_> = commands
                    .into_iter()
                    .map(|cmd| self.resolve_command(cmd))
                    .collect();
                let buffer = wgc::command::CommandBuffer::from_trace(device, resolved_commands);
                queue.submit(&[buffer]).unwrap();
            }
            Action::FailedCommands {
                commands,
                failed_at_submit,
                error,
            } => {
                let action = if failed_at_submit.is_some() {
                    "submitting"
                } else {
                    "encoding"
                };
                if let Some(commands) = commands {
                    log::trace!(
                        "Trace recorded an error {action} the following commands: {commands:#?}"
                    );
                }
                panic!("Error recorded in trace: {error}");
            }
            Action::CreateBlas { id, desc, sizes } => {
                let blas = device.create_blas(&desc, sizes).expect("create_blas error");
                self.blas_s.insert(id, blas);
            }
            Action::DestroyBlas(id) => {
                self.blas_s.remove(&id).expect("invalid blas");
            }
            Action::CreateTlas { id, desc } => {
                let tlas = device.create_tlas(&desc).expect("create_tlas error");
                self.tlas_s.insert(id, tlas);
            }
            Action::DestroyTlas(id) => {
                self.tlas_s.remove(&id).expect("invalid tlas");
            }
        }
    }

    // This one is a little strange because the surface is held by the
    // `player` application but we want to insert the texture into our
    // map so we can find it for rendering.
    pub fn get_surface_texture(
        &mut self,
        id: wgc::id::PointerId<wgc::id::markers::Texture>,
        surface: &wgc::instance::Surface,
    ) {
        let frame = surface
            .get_current_texture()
            .expect("get_current_texture error");
        let texture = frame.texture.expect("did not obtain a surface texture");
        self.textures.insert(id, texture);
    }

    pub fn resolve_buffer_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::Buffer>,
    ) -> Arc<wgc::resource::Buffer> {
        self.buffers.get(&id).expect("invalid buffer").clone()
    }

    fn resolve_texture_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::Texture>,
    ) -> Arc<wgc::resource::Texture> {
        self.textures.get(&id).expect("invalid texture").clone()
    }

    fn resolve_texture_view_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::TextureView>,
    ) -> Arc<wgc::resource::TextureView> {
        self.texture_views
            .get(&id)
            .expect("invalid texture view")
            .clone()
    }

    fn resolve_external_texture_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::ExternalTexture>,
    ) -> Arc<wgc::resource::ExternalTexture> {
        self.external_textures
            .get(&id)
            .expect("invalid external texture")
            .clone()
    }

    fn resolve_sampler_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::Sampler>,
    ) -> Arc<wgc::resource::Sampler> {
        self.samplers.get(&id).expect("invalid sampler").clone()
    }

    fn resolve_bind_group_layout_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::BindGroupLayout>,
    ) -> Arc<wgc::binding_model::BindGroupLayout> {
        self.bind_group_layouts
            .get(&id)
            .expect("invalid bind group layout")
            .clone()
    }

    fn resolve_bind_group_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::BindGroup>,
    ) -> Arc<wgc::binding_model::BindGroup> {
        self.bind_groups
            .get(&id)
            .expect("invalid bind group")
            .clone()
    }

    fn resolve_pipeline_layout_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::PipelineLayout>,
    ) -> Arc<wgc::binding_model::PipelineLayout> {
        self.pipeline_layouts
            .get(&id)
            .expect("invalid pipeline layout")
            .clone()
    }

    fn resolve_shader_module_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::ShaderModule>,
    ) -> Arc<wgc::pipeline::ShaderModule> {
        self.shader_modules
            .get(&id)
            .expect("invalid shader module")
            .clone()
    }

    fn resolve_render_pipeline_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::RenderPipeline>,
    ) -> Arc<wgc::pipeline::RenderPipeline> {
        self.render_pipelines
            .get(&id)
            .expect("invalid render pipeline")
            .clone()
    }

    fn resolve_compute_pipeline_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::ComputePipeline>,
    ) -> Arc<wgc::pipeline::ComputePipeline> {
        self.compute_pipelines
            .get(&id)
            .expect("invalid compute pipeline")
            .clone()
    }

    fn resolve_pipeline_cache_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::PipelineCache>,
    ) -> Arc<wgc::pipeline::PipelineCache> {
        self.pipeline_caches
            .get(&id)
            .expect("invalid pipeline cache")
            .clone()
    }

    fn resolve_render_bundle_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::RenderBundle>,
    ) -> Arc<wgc::command::RenderBundle> {
        self.render_bundles
            .get(&id)
            .expect("invalid render bundle")
            .clone()
    }

    fn resolve_query_set_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::QuerySet>,
    ) -> Arc<wgc::resource::QuerySet> {
        self.query_sets.get(&id).expect("invalid query set").clone()
    }

    fn resolve_blas_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::Blas>,
    ) -> Arc<wgc::resource::Blas> {
        self.blas_s.get(&id).expect("invalid blas").clone()
    }

    fn resolve_tlas_id(
        &self,
        id: wgc::id::PointerId<wgc::id::markers::Tlas>,
    ) -> Arc<wgc::resource::Tlas> {
        self.tlas_s.get(&id).expect("invalid tlas").clone()
    }

    fn resolve_texel_copy_texture_info(
        &self,
        info: wgt::TexelCopyTextureInfo<wgc::id::PointerId<wgc::id::markers::Texture>>,
    ) -> wgt::TexelCopyTextureInfo<Arc<wgc::resource::Texture>> {
        wgt::TexelCopyTextureInfo {
            texture: self.resolve_texture_id(info.texture),
            mip_level: info.mip_level,
            origin: info.origin,
            aspect: info.aspect,
        }
    }

    fn resolve_compute_pipeline_descriptor<'a>(
        &self,
        desc: wgc::device::trace::TraceComputePipelineDescriptor<'a>,
    ) -> wgc::pipeline::ResolvedComputePipelineDescriptor<'a> {
        wgc::pipeline::ResolvedComputePipelineDescriptor {
            label: desc.label,
            layout: desc.layout.map(|id| self.resolve_pipeline_layout_id(id)),
            stage: wgc::pipeline::ResolvedProgrammableStageDescriptor {
                module: self.resolve_shader_module_id(desc.stage.module),
                entry_point: desc.stage.entry_point,
                constants: desc.stage.constants,
                zero_initialize_workgroup_memory: desc.stage.zero_initialize_workgroup_memory,
            },
            cache: desc.cache.map(|id| self.resolve_pipeline_cache_id(id)),
        }
    }

    fn resolve_render_pipeline_descriptor<'a>(
        &self,
        desc: wgc::device::trace::TraceGeneralRenderPipelineDescriptor<'a>,
    ) -> wgc::pipeline::ResolvedGeneralRenderPipelineDescriptor<'a> {
        let layout = desc.layout.map(|id| self.resolve_pipeline_layout_id(id));

        let vertex = match desc.vertex {
            wgc::pipeline::RenderPipelineVertexProcessor::Vertex(vertex_state) => {
                wgc::pipeline::RenderPipelineVertexProcessor::Vertex(
                    wgc::pipeline::ResolvedVertexState {
                        stage: wgc::pipeline::ResolvedProgrammableStageDescriptor {
                            module: self.resolve_shader_module_id(vertex_state.stage.module),
                            entry_point: vertex_state.stage.entry_point,
                            constants: vertex_state.stage.constants,
                            zero_initialize_workgroup_memory: vertex_state
                                .stage
                                .zero_initialize_workgroup_memory,
                        },
                        buffers: vertex_state.buffers,
                    },
                )
            }
            wgc::pipeline::RenderPipelineVertexProcessor::Mesh(task_state, mesh_state) => {
                let resolved_task = task_state.map(|task| wgc::pipeline::ResolvedTaskState {
                    stage: wgc::pipeline::ResolvedProgrammableStageDescriptor {
                        module: self.resolve_shader_module_id(task.stage.module),
                        entry_point: task.stage.entry_point,
                        constants: task.stage.constants,
                        zero_initialize_workgroup_memory: task
                            .stage
                            .zero_initialize_workgroup_memory,
                    },
                });
                let resolved_mesh = wgc::pipeline::ResolvedMeshState {
                    stage: wgc::pipeline::ResolvedProgrammableStageDescriptor {
                        module: self.resolve_shader_module_id(mesh_state.stage.module),
                        entry_point: mesh_state.stage.entry_point,
                        constants: mesh_state.stage.constants,
                        zero_initialize_workgroup_memory: mesh_state
                            .stage
                            .zero_initialize_workgroup_memory,
                    },
                };
                wgc::pipeline::RenderPipelineVertexProcessor::Mesh(resolved_task, resolved_mesh)
            }
        };

        let fragment = desc
            .fragment
            .map(|fragment_state| wgc::pipeline::ResolvedFragmentState {
                stage: wgc::pipeline::ResolvedProgrammableStageDescriptor {
                    module: self.resolve_shader_module_id(fragment_state.stage.module),
                    entry_point: fragment_state.stage.entry_point,
                    constants: fragment_state.stage.constants,
                    zero_initialize_workgroup_memory: fragment_state
                        .stage
                        .zero_initialize_workgroup_memory,
                },
                targets: fragment_state.targets,
            });

        wgc::pipeline::ResolvedGeneralRenderPipelineDescriptor {
            label: desc.label,
            layout,
            vertex,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil,
            multisample: desc.multisample,
            fragment,
            multiview_mask: desc.multiview_mask,
            cache: desc.cache.map(|id| self.resolve_pipeline_cache_id(id)),
        }
    }

    fn resolve_bind_group_descriptor<'a>(
        &self,
        desc: wgc::device::trace::TraceBindGroupDescriptor<'a>,
    ) -> wgc::binding_model::ResolvedBindGroupDescriptor<'a> {
        let layout = self.resolve_bind_group_layout_id(desc.layout);

        let entries: Vec<wgc::binding_model::ResolvedBindGroupEntry> = desc
            .entries
            .to_vec()
            .into_iter()
            .map(|entry| {
                let resource = match entry.resource {
                    BindingResource::Buffer(buffer_binding) => {
                        let buffer = self.resolve_buffer_id(buffer_binding.buffer);
                        wgc::binding_model::ResolvedBindingResource::Buffer(
                            wgc::binding_model::ResolvedBufferBinding {
                                buffer,
                                offset: buffer_binding.offset,
                                size: buffer_binding.size,
                            },
                        )
                    }
                    BindingResource::BufferArray(buffer_bindings) => {
                        let resolved_buffers: Vec<_> = buffer_bindings
                            .to_vec()
                            .into_iter()
                            .map(|bb| {
                                let buffer = self.resolve_buffer_id(bb.buffer);
                                wgc::binding_model::ResolvedBufferBinding {
                                    buffer,
                                    offset: bb.offset,
                                    size: bb.size,
                                }
                            })
                            .collect();
                        wgc::binding_model::ResolvedBindingResource::BufferArray(Cow::Owned(
                            resolved_buffers,
                        ))
                    }
                    BindingResource::Sampler(sampler_id) => {
                        let sampler = self.resolve_sampler_id(sampler_id);
                        wgc::binding_model::ResolvedBindingResource::Sampler(sampler)
                    }
                    BindingResource::SamplerArray(sampler_ids) => {
                        let resolved_samplers: Vec<_> = sampler_ids
                            .to_vec()
                            .into_iter()
                            .map(|id| self.resolve_sampler_id(id))
                            .collect();
                        wgc::binding_model::ResolvedBindingResource::SamplerArray(Cow::Owned(
                            resolved_samplers,
                        ))
                    }
                    BindingResource::TextureView(texture_view_id) => {
                        let texture_view = self.resolve_texture_view_id(texture_view_id);
                        wgc::binding_model::ResolvedBindingResource::TextureView(texture_view)
                    }
                    BindingResource::TextureViewArray(texture_view_ids) => {
                        let resolved_views: Vec<_> = texture_view_ids
                            .to_vec()
                            .into_iter()
                            .map(|id| self.resolve_texture_view_id(id))
                            .collect();
                        wgc::binding_model::ResolvedBindingResource::TextureViewArray(Cow::Owned(
                            resolved_views,
                        ))
                    }
                    BindingResource::AccelerationStructure(tlas_id) => {
                        let tlas = self.resolve_tlas_id(tlas_id);
                        wgc::binding_model::ResolvedBindingResource::AccelerationStructure(tlas)
                    }
                    BindingResource::ExternalTexture(external_texture_id) => {
                        let external_texture =
                            self.resolve_external_texture_id(external_texture_id);
                        wgc::binding_model::ResolvedBindingResource::ExternalTexture(
                            external_texture,
                        )
                    }
                };

                wgc::binding_model::ResolvedBindGroupEntry {
                    binding: entry.binding,
                    resource,
                }
            })
            .collect();

        wgc::binding_model::ResolvedBindGroupDescriptor {
            label: desc.label.clone(),
            layout,
            entries: entries.into(),
        }
    }

    fn resolve_command(&self, command: Command<PointerReferences>) -> ArcCommand {
        match command {
            Command::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => Command::CopyBufferToBuffer {
                src: self.resolve_buffer_id(src),
                src_offset,
                dst: self.resolve_buffer_id(dst),
                dst_offset,
                size,
            },
            Command::CopyBufferToTexture { src, dst, size } => Command::CopyBufferToTexture {
                src: self.resolve_texel_copy_buffer_info(src),
                dst: self.resolve_texel_copy_texture_info(dst),
                size,
            },
            Command::CopyTextureToBuffer { src, dst, size } => Command::CopyTextureToBuffer {
                src: self.resolve_texel_copy_texture_info(src),
                dst: self.resolve_texel_copy_buffer_info(dst),
                size,
            },
            Command::CopyTextureToTexture { src, dst, size } => Command::CopyTextureToTexture {
                src: self.resolve_texel_copy_texture_info(src),
                dst: self.resolve_texel_copy_texture_info(dst),
                size,
            },
            Command::ClearBuffer { dst, offset, size } => Command::ClearBuffer {
                dst: self.resolve_buffer_id(dst),
                offset,
                size,
            },
            Command::ClearTexture {
                dst,
                subresource_range,
            } => Command::ClearTexture {
                dst: self.resolve_texture_id(dst),
                subresource_range,
            },
            Command::WriteTimestamp {
                query_set,
                query_index,
            } => Command::WriteTimestamp {
                query_set: self.resolve_query_set_id(query_set),
                query_index,
            },
            Command::ResolveQuerySet {
                query_set,
                start_query,
                query_count,
                destination,
                destination_offset,
            } => Command::ResolveQuerySet {
                query_set: self.resolve_query_set_id(query_set),
                start_query,
                query_count,
                destination: self.resolve_buffer_id(destination),
                destination_offset,
            },
            Command::PushDebugGroup(label) => Command::PushDebugGroup(label.clone()),
            Command::PopDebugGroup => Command::PopDebugGroup,
            Command::InsertDebugMarker(label) => Command::InsertDebugMarker(label.clone()),
            Command::RunComputePass {
                pass,
                timestamp_writes,
            } => Command::RunComputePass {
                pass: self.resolve_compute_pass(pass),
                timestamp_writes: timestamp_writes.map(|tw| self.resolve_pass_timestamp_writes(tw)),
            },
            Command::RunRenderPass {
                pass,
                color_attachments,
                depth_stencil_attachment,
                timestamp_writes,
                occlusion_query_set,
                multiview_mask,
            } => Command::RunRenderPass {
                pass: self.resolve_render_pass(pass),
                color_attachments: self.resolve_color_attachments(color_attachments),
                depth_stencil_attachment: depth_stencil_attachment
                    .map(|att| self.resolve_depth_stencil_attachment(att)),
                timestamp_writes: timestamp_writes.map(|tw| self.resolve_pass_timestamp_writes(tw)),
                occlusion_query_set: occlusion_query_set.map(|qs| self.resolve_query_set_id(qs)),
                multiview_mask,
            },
            Command::BuildAccelerationStructures { blas, tlas } => {
                Command::BuildAccelerationStructures {
                    blas: blas
                        .into_iter()
                        .map(|entry| self.resolve_blas_build_entry(entry))
                        .collect(),
                    tlas: tlas
                        .into_iter()
                        .map(|package| self.resolve_tlas_package(package))
                        .collect(),
                }
            }
            Command::TransitionResources {
                buffer_transitions,
                texture_transitions,
            } => Command::TransitionResources {
                buffer_transitions: buffer_transitions
                    .into_iter()
                    .map(|trans| self.resolve_buffer_transition(trans))
                    .collect(),
                texture_transitions: texture_transitions
                    .into_iter()
                    .map(|trans| self.resolve_texture_transition(trans))
                    .collect(),
            },
        }
    }

    // Helper methods for command resolution
    fn resolve_texel_copy_buffer_info(
        &self,
        info: wgt::TexelCopyBufferInfo<PointerId<wgc::id::markers::Buffer>>,
    ) -> wgt::TexelCopyBufferInfo<Arc<wgc::resource::Buffer>> {
        wgt::TexelCopyBufferInfo {
            buffer: self
                .buffers
                .get(&info.buffer)
                .cloned()
                .expect("invalid buffer"),
            layout: info.layout,
        }
    }

    fn resolve_compute_pass(
        &self,
        pass: BasePass<wgc::command::ComputeCommand<PointerReferences>, Infallible>,
    ) -> BasePass<wgc::command::ComputeCommand<ArcReferences>, Infallible> {
        let BasePass {
            label,
            error,
            commands,
            dynamic_offsets,
            immediates_data,
            string_data,
        } = pass;

        BasePass {
            label,
            error,
            commands: commands
                .into_iter()
                .map(|cmd| self.resolve_compute_command(cmd))
                .collect(),
            dynamic_offsets,
            immediates_data,
            string_data,
        }
    }

    fn resolve_render_pass(
        &self,
        pass: BasePass<wgc::command::RenderCommand<PointerReferences>, Infallible>,
    ) -> BasePass<wgc::command::RenderCommand<ArcReferences>, Infallible> {
        let BasePass {
            label,
            error,
            commands,
            dynamic_offsets,
            immediates_data,
            string_data,
        } = pass;

        BasePass {
            label,
            error,
            commands: commands
                .into_iter()
                .map(|cmd| self.resolve_render_command(cmd))
                .collect(),
            dynamic_offsets,
            immediates_data,
            string_data,
        }
    }

    fn resolve_compute_command(
        &self,
        command: wgc::command::ComputeCommand<PointerReferences>,
    ) -> wgc::command::ComputeCommand<ArcReferences> {
        use wgc::command::ComputeCommand as C;
        match command {
            C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group: bind_group.map(|bg| self.resolve_bind_group_id(bg)),
            },
            C::SetPipeline(id) => C::SetPipeline(self.resolve_compute_pipeline_id(id)),
            C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            },
            C::Dispatch(groups) => C::Dispatch(groups),
            C::DispatchIndirect { buffer, offset } => C::DispatchIndirect {
                buffer: self.resolve_buffer_id(buffer),
                offset,
            },
            C::PushDebugGroup { color, len } => C::PushDebugGroup { color, len },
            C::PopDebugGroup => C::PopDebugGroup,
            C::InsertDebugMarker { color, len } => C::InsertDebugMarker { color, len },
            C::WriteTimestamp {
                query_set,
                query_index,
            } => C::WriteTimestamp {
                query_set: self.resolve_query_set_id(query_set),
                query_index,
            },
            C::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => C::BeginPipelineStatisticsQuery {
                query_set: self.resolve_query_set_id(query_set),
                query_index,
            },
            C::EndPipelineStatisticsQuery => C::EndPipelineStatisticsQuery,
        }
    }

    fn resolve_render_command(
        &self,
        command: wgc::command::RenderCommand<PointerReferences>,
    ) -> wgc::command::RenderCommand<ArcReferences> {
        use wgc::command::RenderCommand as C;
        match command {
            C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group: bind_group.map(|bg| self.resolve_bind_group_id(bg)),
            },
            C::SetPipeline(id) => C::SetPipeline(self.resolve_render_pipeline_id(id)),
            C::SetIndexBuffer {
                buffer,
                index_format,
                offset,
                size,
            } => C::SetIndexBuffer {
                buffer: self.resolve_buffer_id(buffer),
                index_format,
                offset,
                size,
            },
            C::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => C::SetVertexBuffer {
                slot,
                buffer: self.resolve_buffer_id(buffer),
                offset,
                size,
            },
            C::SetBlendConstant(color) => C::SetBlendConstant(color),
            C::SetStencilReference(val) => C::SetStencilReference(val),
            C::SetViewport {
                rect,
                depth_min,
                depth_max,
            } => C::SetViewport {
                rect,
                depth_min,
                depth_max,
            },
            C::SetScissor(rect) => C::SetScissor(rect),
            C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            },
            C::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => C::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            },
            C::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => C::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            },
            C::DrawMeshTasks {
                group_count_x,
                group_count_y,
                group_count_z,
            } => C::DrawMeshTasks {
                group_count_x,
                group_count_y,
                group_count_z,
            },
            C::DrawIndirect {
                buffer,
                offset,
                count,
                family,
                vertex_or_index_limit,
                instance_limit,
            } => C::DrawIndirect {
                buffer: self.resolve_buffer_id(buffer),
                offset,
                count,
                family,
                vertex_or_index_limit,
                instance_limit,
            },
            C::MultiDrawIndirectCount {
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_count,
                family,
            } => C::MultiDrawIndirectCount {
                buffer: self.resolve_buffer_id(buffer),
                offset,
                count_buffer: self.resolve_buffer_id(count_buffer),
                count_buffer_offset,
                max_count,
                family,
            },
            C::PushDebugGroup { color, len } => C::PushDebugGroup { color, len },
            C::PopDebugGroup => C::PopDebugGroup,
            C::InsertDebugMarker { color, len } => C::InsertDebugMarker { color, len },
            C::WriteTimestamp {
                query_set,
                query_index,
            } => C::WriteTimestamp {
                query_set: self.resolve_query_set_id(query_set),
                query_index,
            },
            C::BeginOcclusionQuery { query_index } => C::BeginOcclusionQuery { query_index },
            C::EndOcclusionQuery => C::EndOcclusionQuery,
            C::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => C::BeginPipelineStatisticsQuery {
                query_set: self.resolve_query_set_id(query_set),
                query_index,
            },
            C::EndPipelineStatisticsQuery => C::EndPipelineStatisticsQuery,
            C::ExecuteBundle(bundle) => C::ExecuteBundle(self.resolve_render_bundle_id(bundle)),
        }
    }

    fn resolve_pass_timestamp_writes(
        &self,
        writes: wgc::command::PassTimestampWrites<PointerId<wgc::id::markers::QuerySet>>,
    ) -> wgc::command::PassTimestampWrites<Arc<wgc::resource::QuerySet>> {
        wgc::command::PassTimestampWrites {
            query_set: self.resolve_query_set_id(writes.query_set),
            beginning_of_pass_write_index: writes.beginning_of_pass_write_index,
            end_of_pass_write_index: writes.end_of_pass_write_index,
        }
    }

    fn resolve_color_attachments(
        &self,
        attachments: wgc::command::ColorAttachments<PointerId<wgc::id::markers::TextureView>>,
    ) -> wgc::command::ColorAttachments<Arc<wgc::resource::TextureView>> {
        attachments
            .into_iter()
            .map(|opt| {
                opt.map(|att| wgc::command::RenderPassColorAttachment {
                    view: self.resolve_texture_view_id(att.view),
                    depth_slice: att.depth_slice,
                    resolve_target: att
                        .resolve_target
                        .map(|rt| self.resolve_texture_view_id(rt)),
                    load_op: att.load_op,
                    store_op: att.store_op,
                })
            })
            .collect()
    }

    fn resolve_depth_stencil_attachment(
        &self,
        attachment: wgc::command::ResolvedRenderPassDepthStencilAttachment<
            PointerId<wgc::id::markers::TextureView>,
        >,
    ) -> wgc::command::ResolvedRenderPassDepthStencilAttachment<Arc<wgc::resource::TextureView>>
    {
        wgc::command::ResolvedRenderPassDepthStencilAttachment {
            view: self.resolve_texture_view_id(attachment.view),
            depth: attachment.depth,
            stencil: attachment.stencil,
        }
    }

    fn resolve_blas_build_entry(
        &self,
        entry: wgc::ray_tracing::OwnedBlasBuildEntry<PointerReferences>,
    ) -> wgc::ray_tracing::OwnedBlasBuildEntry<ArcReferences> {
        wgc::ray_tracing::OwnedBlasBuildEntry {
            blas: self.resolve_blas_id(entry.blas),
            geometries: self.resolve_blas_geometries(entry.geometries),
        }
    }

    fn resolve_tlas_package(
        &self,
        package: wgc::ray_tracing::OwnedTlasPackage<PointerReferences>,
    ) -> wgc::ray_tracing::OwnedTlasPackage<ArcReferences> {
        wgc::ray_tracing::OwnedTlasPackage {
            tlas: self.resolve_tlas_id(package.tlas),
            instances: package
                .instances
                .into_iter()
                .map(|opt| opt.map(|inst| self.resolve_tlas_instance(inst)))
                .collect(),
            lowest_unmodified: package.lowest_unmodified,
        }
    }

    // Helper functions for ray tracing structures
    fn resolve_blas_geometries(
        &self,
        geometries: wgc::ray_tracing::OwnedBlasGeometries<PointerReferences>,
    ) -> wgc::ray_tracing::OwnedBlasGeometries<ArcReferences> {
        match geometries {
            wgc::ray_tracing::OwnedBlasGeometries::TriangleGeometries(geos) => {
                wgc::ray_tracing::OwnedBlasGeometries::TriangleGeometries(
                    geos.into_iter()
                        .map(|geo| self.resolve_blas_triangle_geometry(geo))
                        .collect(),
                )
            }
        }
    }

    fn resolve_blas_triangle_geometry(
        &self,
        geometry: wgc::ray_tracing::OwnedBlasTriangleGeometry<PointerReferences>,
    ) -> wgc::ray_tracing::OwnedBlasTriangleGeometry<ArcReferences> {
        wgc::ray_tracing::OwnedBlasTriangleGeometry {
            size: geometry.size,
            vertex_buffer: self.resolve_buffer_id(geometry.vertex_buffer),
            index_buffer: geometry.index_buffer.map(|buf| self.resolve_buffer_id(buf)),
            transform_buffer: geometry
                .transform_buffer
                .map(|buf| self.resolve_buffer_id(buf)),
            first_vertex: geometry.first_vertex,
            vertex_stride: geometry.vertex_stride,
            first_index: geometry.first_index,
            transform_buffer_offset: geometry.transform_buffer_offset,
        }
    }

    fn resolve_tlas_instance(
        &self,
        instance: wgc::ray_tracing::OwnedTlasInstance<PointerReferences>,
    ) -> wgc::ray_tracing::OwnedTlasInstance<ArcReferences> {
        wgc::ray_tracing::OwnedTlasInstance {
            blas: self.resolve_blas_id(instance.blas),
            transform: instance.transform,
            custom_data: instance.custom_data,
            mask: instance.mask,
        }
    }

    fn resolve_buffer_transition(
        &self,
        trans: wgt::BufferTransition<PointerId<wgc::id::markers::Buffer>>,
    ) -> wgt::BufferTransition<Arc<wgc::resource::Buffer>> {
        wgt::BufferTransition {
            buffer: self
                .buffers
                .get(&trans.buffer)
                .cloned()
                .expect("invalid buffer"),
            state: trans.state,
        }
    }

    fn resolve_texture_transition(
        &self,
        trans: wgt::TextureTransition<PointerId<wgc::id::markers::Texture>>,
    ) -> wgt::TextureTransition<Arc<wgc::resource::Texture>> {
        wgt::TextureTransition {
            texture: self
                .textures
                .get(&trans.texture)
                .cloned()
                .expect("invalid texture"),
            selector: trans.selector.clone(),
            state: trans.state,
        }
    }
}
