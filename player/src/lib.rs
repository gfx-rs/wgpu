/*! This is a player library for WebGPU traces.
 *
 * # Notes
 * - we call device_maintain_ids() before creating any refcounted resource,
 *   which is basically everything except for BGL and shader modules,
 *   so that we don't accidentally try to use the same ID.
!*/

use wgc::{id::Cached, device::trace};

use std::{
    borrow::{Borrow, Cow},
    collections::HashSet, convert::TryInto, fmt::Debug, fs, marker::PhantomData, path::Path
};

/// Map used to hold resources temporarily between creation and assignment.
pub type TraceIdMap = wgc::id::IdMap<trace::TraceResourceId, wgc::id::Dummy, wgc::id::IdCon>;

#[derive(Debug)]
pub struct IdentityPassThrough<I>(PhantomData<I>);

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandler<I> for IdentityPassThrough<I> {
    type Input = I;
    fn process(&self, id: I, backend: wgt::Backend) -> I {
        let (index, epoch, _backend) = id.unzip();
        I::zip(index, epoch, backend)
    }
    fn free(&self, _id: I) {}
}

pub struct IdentityPassThroughFactory;

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandlerFactory<I>
    for IdentityPassThroughFactory
{
    type Filter = IdentityPassThrough<I>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityPassThrough(PhantomData)
    }
}
impl wgc::hub::GlobalIdentityHandlerFactory for IdentityPassThroughFactory {}

/// Wrapper around Action to efficiently handle temporary object creation and destruction.
pub enum Action<'a> {
    /// The underlying trace action.
    Trace(trace::Action<'a>),
    /// The destroy action inferred from analysis of the trace.
    Destroy(usize),
}

pub trait GlobalPlay {
    /// Analyze the trace to infer a new trace with resource destruction actions included.
    ///
    /// NOTE: The returned list of actions is reversed, due to the way we analyze from back to
    /// front.  We could reverse again before returning, but it turns out that the player also
    /// wants actions reversed, so it's faster to just leave it as is.
    fn init_actions<'a>(&self, trace: Vec<trace::Action<'a>>) -> Vec<Action<'a>>;
    fn encode_commands<A: wgc::hub::HalApi>(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
        cache: &wgc::id::IdCache2,
    ) -> wgc::id::CommandBufferId;
    fn process<'a: 'b, 'b, A: wgc::hub::HalApi + 'a>(
        &self,
        device: wgc::id::IdGuard<A, wgc::device::Device<wgc::id::Dummy>>,
        action: Action<'a>,
        dir: &Path,
        trace_cache: &'b mut TraceIdMap,
        cache: &'b mut wgc::id::IdCache2,
        comb_manager: &mut wgc::hub::IdentityManager,
    );
            // wgc::id::Id2<wgc::binding_model::BindGroup<wgc::api::Empty>>: wgc::id::AllBackends<A>;
}

impl GlobalPlay for wgc::hub::Global<IdentityPassThroughFactory> {
    fn init_actions<'a>(&self, trace: Vec<trace::Action<'a>>) -> Vec<Action<'a>> {
        log::info!("init_actions");

        // The set of currently active resource ids, added to the set on their final use in the
        // trace and removed from the set on their initial assignment.
        let mut active = HashSet::<usize>::new();
        let mut actions = Vec::with_capacity(trace.len());
        // We iterate backwards in order to find the last use of each command.
        trace.into_iter().rev().for_each(|action| {
            use wgc::device::trace::Action::*;

            let mut find_last_use = |id: Cached<wgc::id::Dummy, &wgc::id::UsizeCon>| ->
                Result<(), core::convert::Infallible> {
                let id = *id.borrow();
                if active.insert(id) {
                    actions.push(Action::Destroy(id));
                }
                Ok(())
            };

            // Analayze the latest event that hasn't been analyzed yet to look for references to
            // resources we haven't seen yet, and insert destroy events immediately after them in
            // the action list.  Note that since we're iterating backwards, we actually push the
            // destroy onto the vector *before* the last action where it's used, then reverse the
            // actions list at the end.
            match &action {
                // Handled elsewhere.
                Init { .. } | ConfigureSurface { .. } | Present (_) => {},
                // A read of a resource that's used later works exactly the same as an Assign to
                // the returned resource (resource_id) plus a use of the resource we read from
                // (pipeline_id).  Because pipeline_id keeps a strong reference to resource_id,
                // this is sound even if resource_id was previously created by us.  If the resource
                // is not used later, we simply skip this action (we could also perform this
                // optimization with Create/Assign pairs, but it's a little harder to do because
                // they're split in two).
                GetComputePipelineBindGroupLayout { pipeline_id, resource_id, .. }
                | GetRenderPipelineBindGroupLayout { pipeline_id, resource_id, .. } =>
                    if active.remove(resource_id) {
                        if active.insert(*pipeline_id) {
                            actions.push(Action::Destroy(*pipeline_id));
                        }
                    } else {
                        return;
                    },
                // Will go away soon (deletions).
                | DestroyBuffer(_)
                | DestroyTexture(_)
                => {},
                // Resource creation.
                //
                // Note that when we see an assignment for an element that was never used,
                // we know that it was successfully created (or else the assignment wouldn't be
                // in the crash log).  Since it didn't crash on creation, it should be safe
                // to remove the insertion, so in theory we could avoid counting this as a use of
                // the id and omit creating the resource in the first place.  But for now, for
                // simplicity, we choose to treat this as a use.
                Assign { resource_id, .. } => if !active.remove(resource_id) {
                    actions.push(Action::Destroy(*resource_id));
                },
                CreateBuffer(_id, _desc) => {},
                FreeBuffer(_id) =>
                    /*find_last_use(Cached::Buffer(id)).unwrap_or_else(|e| match e {})*/{},
                CreateTexture(_id, _desc) => {},
                FreeTexture(_id) =>
                    /*find_last_use(Cached::Texture(id)).unwrap_or_else(|e| match e {})*/{},
                CreateTextureView { desc, parent_id: _, .. } => {
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {});
                    // FIXME: Uncomment when Texture is Arc'd.
                    // find_last_use(&parent_id).unwrap_or_else(|e| match e {});
                },
                CreateSampler(_id, desc) =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                GetSurfaceTexture { id: _, parent_id: _ } => {
                    // FIXME: Uncomment when Texture is Arc'd.
                    // find_last_use(Cached::Texture(id)).unwrap_or_else(|e| match e {});
                    // // FIXME: Uncomment when Surface is Arc'd.
                    // // find_last_use(Cached::Surface(parent_id)).unwrap_or_else(|e| match e {});
                },
                CreateBindGroupLayout(_id, desc) =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                CreatePipelineLayout(_id, desc) =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                CreateBindGroup(_id, desc) =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                CreateShaderModule { desc, .. } =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                CreateComputePipeline { desc, .. } =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                CreateRenderPipeline { desc, .. } =>
                    desc.trace_resources(find_last_use).unwrap_or_else(|e| match e {}),
                CreateRenderBundle { base, .. } =>
                    base.commands.iter().for_each(|command| {
                        command.trace_resources(&mut find_last_use).unwrap_or_else(|e| match e {});
                    }),
                CreateQuerySet { .. } => {},
                WriteBuffer { id: _, .. } =>
                    /*{ find_last_use(Cached::Buffer(id)).unwrap_or_else(|e| match e {}); }*/{},
                WriteTexture { to: _, .. } =>
                    /*{ find_last_use(Cached::Texture(to.texture)).unwrap_or_else(|e| match e {}); }*/{},
                Submit(_index, commands) => commands.iter().for_each(|command| {
                    command.trace_resources(&mut find_last_use).unwrap_or_else(|e| match e {});
                }),
            }

            actions.push(Action::Trace(action));
        });
        actions
    }

    fn encode_commands<A: wgc::hub::HalApi>(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
        cache: &wgc::id::IdCache2,
    ) -> wgc::id::CommandBufferId {
        for command in commands {
            match command {
                trace::Command::CopyBufferToBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => self
                    .command_encoder_copy_buffer_to_buffer::<A>(
                        encoder, src, src_offset, dst, dst_offset, size,
                    )
                    .unwrap(),
                trace::Command::CopyBufferToTexture { src, dst, size } => self
                    .command_encoder_copy_buffer_to_texture::<A>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToBuffer { src, dst, size } => self
                    .command_encoder_copy_texture_to_buffer::<A>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToTexture { src, dst, size } => self
                    .command_encoder_copy_texture_to_texture::<A>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::ClearBuffer { dst, offset, size } => self
                    .command_encoder_clear_buffer::<A>(encoder, dst, offset, size)
                    .unwrap(),
                trace::Command::ClearImage {
                    dst,
                    subresource_range,
                } => self
                    .command_encoder_clear_image::<A>(encoder, dst, &subresource_range)
                    .unwrap(),
                trace::Command::WriteTimestamp {
                    ref query_set_id,
                    query_index,
                } => self
                    .command_encoder_write_timestamp::<A>(encoder, cache.get::<wgc::resource::QuerySet<wgc::api::Empty>>(query_set_id).unwrap(), query_index)
                    .unwrap(),
                trace::Command::ResolveQuerySet {
                    ref query_set_id,
                    start_query,
                    query_count,
                    destination,
                    destination_offset,
                } => self
                    .command_encoder_resolve_query_set::<A>(
                        encoder,
                        cache.get::<wgc::resource::QuerySet<wgc::api::Empty>>(query_set_id).unwrap(),
                        start_query,
                        query_count,
                        destination,
                        destination_offset,
                    )
                    .unwrap(),
                trace::Command::RunComputePass { base } => {
                    self.command_encoder_run_compute_pass_impl::<A>(encoder, base.try_from_owned(cache).unwrap().as_ref())
                        .unwrap();
                }
                trace::Command::RunRenderPass {
                    base,
                    target_colors,
                    target_depth_stencil,
                } => {
                    let target_depth_stencil = target_depth_stencil.map(|id| (cache, id).try_into().unwrap());
                    let target_colors: Vec<_> = target_colors.iter().map(|&at| (cache, at).try_into().unwrap()).collect();
                    self.command_encoder_run_render_pass_impl::<A>(
                        encoder,
                        base.try_from_owned(cache).unwrap().as_ref(),
                        &target_colors,
                        target_depth_stencil.as_ref(),
                    )
                    .unwrap();
                }
            }
        }
        let (cmd_buf, error) = self
            .command_encoder_finish::<A>(encoder, &wgt::CommandBufferDescriptor { label: None });
        if let Some(e) = error {
            panic!("{:?}", e);
        }
        cmd_buf
    }

    fn process<'a: 'b, 'b, A: wgc::hub::HalApi + 'a>(
        &self,
        device: wgc::id::IdGuard<A, wgc::device::Device<wgc::id::Dummy>>,
        action: Action<'a>,
        dir: &Path,
        trace_cache: &'b mut TraceIdMap,
        cache: &'b mut wgc::id::IdCache2,
        comb_manager: &mut wgc::hub::IdentityManager,
    ) {
        use crate::Action as PlayAction;
        use wgc::device::trace::Action;

        let action = match action {
            PlayAction::Destroy(id) => {
                let resource = cache.destroy(&id);
                log::info!("Destroy: {:?} (resource_id={:?})", resource, id);
                return;
            },
            PlayAction::Trace(action) => action,
        };

        log::info!("action {:?}", action);
        //TODO: find a way to force ID perishing without excessive `maintain()` calls.
        match action {
            Action::Init { .. } => {
                panic!("Unexpected Action::Init: has to be the first action only")
            }
            Action::ConfigureSurface { .. } | Action::Present(_) => {
                panic!("Unexpected Surface action: winit feature is not enabled")
            }
            Action::Assign { trace_id, resource_id } => {
                let resource = trace_cache.destroy(&trace_id);
                log::info!("Assign: {:?} (resource_id={:?}, trace_id={:?}), ", resource, resource_id, trace_id);
                assert_eq!(
                    cache.insert(resource_id, resource),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            Action::CreateBuffer(id, desc) => {
                self.device_maintain_ids::<A>(device).unwrap();
                let (_, error) = self.device_create_buffer::<A>(device.to_owned(), &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            Action::FreeBuffer(id) => {
                self.buffer_destroy::<A>(id).unwrap();
            }
            Action::DestroyBuffer(id) => {
                self.buffer_drop::<A>(id, true);
            }
            Action::CreateTexture(id, desc) => {
                self.device_maintain_ids::<A>(device).unwrap();
                let (_, error) = self.device_create_texture::<A>(device.to_owned(), &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            Action::FreeTexture(id) => {
                self.texture_destroy::<A>(id).unwrap();
            }
            Action::DestroyTexture(id) => {
                self.texture_drop::<A>(id, true);
            }
            Action::CreateTextureView {
                id,
                parent_id,
                desc,
            } => {
                self.device_maintain_ids::<A>(device).unwrap();
                let texture_view = self.texture_create_view::<A>(parent_id, &desc).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::resource::TextureView<wgc::api::Empty>>(id, texture_view),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /* Action::DestroyTextureView(id) => {
                self.texture_view_drop::<A>(id, true).unwrap();
            } */
            Action::CreateSampler(id, desc) => {
                self.device_maintain_ids::<A>(device).unwrap();
                let sampler = Self::device_create_sampler::<A>(device.to_owned(), &desc).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::resource::Sampler<wgc::api::Empty>>(id, sampler),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /*Action::DestroySampler(id) => {
                let id = cache.destroy::<wgc::resource::Sampler<wgc::api::Empty>>(&id).unwrap();
                let id = wgc::id::expect_backend_owned::<_, A>(id);
                Self::sampler_drop::<A>(id);
            }*/
            Action::GetSurfaceTexture { id, parent_id } => {
                self.device_maintain_ids::<A>(device).unwrap();
                self.surface_get_current_texture::<A>(parent_id, id)
                    .unwrap()
                    .texture_id
                    .unwrap();
            }
            Action::CreateBindGroupLayout(id, desc) => {
                let bind_group_layout = Self::device_create_bind_group_layout::<A>(device.to_owned(), desc).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::binding_model::BindGroupLayout<wgc::api::Empty>>(id, bind_group_layout),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /* Action::DestroyBindGroupLayout(id) => {
                self.bind_group_layout_drop::<A>(id);
            } */
            Action::CreatePipelineLayout(id, desc) => {
                self.device_maintain_ids::<A>(device).unwrap();
                let desc = (&*cache, desc).try_into().unwrap();
                let pipeline_layout = Self::device_create_pipeline_layout::<A>(device.to_owned(), desc).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::binding_model::PipelineLayout<wgc::api::Empty>>(id, pipeline_layout),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /*Action::DestroyPipelineLayout(id) => {
                let id = cache.destroy::<wgc::binding_model::PipelineLayout<wgc::api::Empty>>(&id).unwrap();
                let id = wgc::id::expect_backend_owned::<_, A>(id);
                Self::pipeline_layout_drop::<A>(id);
            }*/
            Action::CreateBindGroup(id, desc) => {
                self.device_maintain_ids::<A>(device).unwrap();
                let layout = TryInto::<wgc::id::IdGuard<_, wgc::binding_model::BindGroupLayout<_>>>::try_into(
                    (&*cache, desc.layout)
                ).unwrap().into();
                let bind_group = self.device_create_bind_group::<A, _>(
                    device.to_owned(),
                    wgc::binding_model::BindGroupDescriptor {
                        label: desc.label.as_deref().map(Cow::Borrowed),
                        layout,
                        entries: desc.entries.iter().map(|entry| (&*cache, entry).try_into().unwrap()),
                    },
                ).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::binding_model::BindGroup<wgc::api::Empty>>(id, bind_group),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /* Action::DestroyBindGroup(id) => {
                let id = cache.destroy::<wgc::binding_model::BindGroup<wgc::api::Empty>>(&id).unwrap();
                let id = wgc::id::expect_backend_owned::<_, A>(id);
                Self::bind_group_drop::<A>(id);
            } */
            Action::CreateShaderModule { id, desc, data } => {
                log::info!("Creating shader from {}", data);
                let code = fs::read_to_string(dir.join(&data)).unwrap();
                let source = if data.ends_with(".wgsl") {
                    wgc::pipeline::ShaderModuleSource::Wgsl(Cow::Owned(code))
                } else if data.ends_with(".ron") {
                    let module = ron::de::from_str(&code).unwrap();
                    wgc::pipeline::ShaderModuleSource::Naga(module)
                } else {
                    panic!("Unknown shader {}", data);
                };
                let shader_module = Self::device_create_shader_module::<A>(device.to_owned(), &desc, source).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::pipeline::ShaderModule<wgc::api::Empty>>(id, shader_module),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /* Action::DestroyShaderModule(id) => {
                let id = cache.destroy::<wgc::pipeline::ShaderModule<wgc::api::Empty>>(&id).unwrap();
                let id = wgc::id::expect_backend_box_owned::<_, A>(id);
                Self::shader_module_drop::<A>(id);
            } */
            Action::CreateComputePipeline {
                id,
                desc,
            } => {
                self.device_maintain_ids::<A>(device).unwrap();
                let desc = (&*cache, desc).try_into().unwrap();
                let pipeline = Self::device_create_compute_pipeline::<A>(device.to_owned(), desc/*, implicit_ids*/).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::pipeline::ComputePipeline<wgc::api::Empty>>(id, pipeline),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            Action::GetComputePipelineBindGroupLayout { pipeline_id, index, resource_id } => {
                let pipeline = (&*cache, pipeline_id).try_into().unwrap();
                let bind_group_layout = Self::compute_pipeline_get_bind_group_layout::<A>(pipeline, index).unwrap();
                assert_eq!(
                    cache.create::<wgc::binding_model::BindGroupLayout<wgc::api::Empty>>(resource_id, bind_group_layout),
                    None,
                    "Id was created twice without being destroyed.",
                );
            },
            /* Action::DestroyComputePipeline(id) => {
                self.compute_pipeline_drop::<A>(id);
            } */
            Action::CreateRenderPipeline {
                id,
                desc,
            } => {
                self.device_maintain_ids::<A>(device).unwrap();
                let desc = (&*cache, desc).try_into().unwrap();
                let pipeline = Self::device_create_render_pipeline::<A>(device.to_owned(), desc/*, implicit_ids*/).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::pipeline::RenderPipeline<wgc::api::Empty>>(id, pipeline),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            Action::GetRenderPipelineBindGroupLayout { pipeline_id, index, resource_id } => {
                let pipeline = (&*cache, pipeline_id).try_into().unwrap();
                let bind_group_layout = Self::render_pipeline_get_bind_group_layout::<A>(pipeline, index).unwrap();
                assert_eq!(
                    cache.create::<wgc::binding_model::BindGroupLayout<wgc::api::Empty>>(resource_id, bind_group_layout),
                    None,
                    "Id was created twice without being destroyed.",
                );
            },
            /* Action::DestroyRenderPipeline(id) => {
                self.render_pipeline_drop::<A>(id);
            } */
            Action::CreateRenderBundle { id, desc, base } => {
                let device_id = wgc::id::Id2::upcast_backend(device.to_owned());
                let bundle =
                    wgc::command::RenderBundleEncoder::new(&desc, &device_id, Some(base.try_from_owned(cache).unwrap())).unwrap();
                let bundle = self.render_bundle_encoder_finish::<A>(
                    bundle,
                    &wgt::RenderBundleDescriptor { label: desc.label },
                ).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::command::RenderBundle<wgc::api::Empty>>(id, bundle),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /* Action::DestroyRenderBundle(id) => {
                self.render_bundle_drop::<A>(id);
            } */
            Action::CreateQuerySet { id, desc } => {
                self.device_maintain_ids::<A>(device).unwrap();
                let query_set = Self::device_create_query_set::<A>(device.to_owned(), &desc).unwrap();
                assert_eq!(
                    trace_cache.create::<wgc::resource::QuerySet<wgc::api::Empty>>(id, query_set),
                    None,
                    "Id was created twice without being destroyed.",
                );
            }
            /* Action::DestroyQuerySet(id) => {
                let id = cache.destroy::<wgc::resource::QuerySet<wgc::api::Empty>>(&id).unwrap();
                let id = wgc::id::expect_backend_owned::<_, A>(id);
                Self::query_set_drop::<A>(id);
            } */
            Action::WriteBuffer {
                id,
                data,
                range,
                queued,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                let size = (range.end - range.start) as usize;
                if queued {
                    self.queue_write_buffer::<A>(device, id, range.start, &bin)
                        .unwrap();
                } else {
                    self.device_wait_for_buffer::<A>(device, id).unwrap();
                    self.device_set_buffer_sub_data::<A>(device, id, range.start, &bin[..size])
                        .unwrap();
                }
            }
            Action::WriteTexture {
                to,
                data,
                layout,
                size,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                self.queue_write_texture::<A>(device, &to, &bin, &layout, &size)
                    .unwrap();
            }
            Action::Submit(_index, ref commands) if commands.is_empty() => {
                self.queue_submit::<A, _>(device, core::iter::empty()).unwrap();
            }
            Action::Submit(_index, commands) => {
                let (encoder, error) = self.device_create_command_encoder::<A>(
                    device.to_owned(),
                    &wgt::CommandEncoderDescriptor { label: None },
                    comb_manager.alloc(A::VARIANT),
                );
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
                let cmdbuf = self.encode_commands::<A>(encoder, commands, cache);
                self.queue_submit::<A, _>(device, core::iter::once(cmdbuf)).unwrap();
            }
        }
    }
}
