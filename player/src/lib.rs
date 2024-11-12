//! This is a player library for WebGPU traces.

#![cfg(not(target_arch = "wasm32"))]
#![warn(unsafe_op_in_unsafe_fn)]

use wgc::device::trace;

use std::{borrow::Cow, fs, path::Path};

pub trait GlobalPlay {
    fn encode_commands(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
    ) -> wgc::id::CommandBufferId;
    fn process(
        &self,
        device: wgc::id::DeviceId,
        queue: wgc::id::QueueId,
        action: trace::Action,
        dir: &Path,
        comb_manager: &mut wgc::identity::IdentityManager<wgc::id::markers::CommandBuffer>,
    );
}

impl GlobalPlay for wgc::global::Global {
    fn encode_commands(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
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
                    .command_encoder_copy_buffer_to_buffer(
                        encoder, src, src_offset, dst, dst_offset, size,
                    )
                    .unwrap(),
                trace::Command::CopyBufferToTexture { src, dst, size } => self
                    .command_encoder_copy_buffer_to_texture(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToBuffer { src, dst, size } => self
                    .command_encoder_copy_texture_to_buffer(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToTexture { src, dst, size } => self
                    .command_encoder_copy_texture_to_texture(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::ClearBuffer { dst, offset, size } => self
                    .command_encoder_clear_buffer(encoder, dst, offset, size)
                    .unwrap(),
                trace::Command::ClearTexture {
                    dst,
                    subresource_range,
                } => self
                    .command_encoder_clear_texture(encoder, dst, &subresource_range)
                    .unwrap(),
                trace::Command::WriteTimestamp {
                    query_set_id,
                    query_index,
                } => self
                    .command_encoder_write_timestamp(encoder, query_set_id, query_index)
                    .unwrap(),
                trace::Command::ResolveQuerySet {
                    query_set_id,
                    start_query,
                    query_count,
                    destination,
                    destination_offset,
                } => self
                    .command_encoder_resolve_query_set(
                        encoder,
                        query_set_id,
                        start_query,
                        query_count,
                        destination,
                        destination_offset,
                    )
                    .unwrap(),
                trace::Command::PushDebugGroup(marker) => self
                    .command_encoder_push_debug_group(encoder, &marker)
                    .unwrap(),
                trace::Command::PopDebugGroup => {
                    self.command_encoder_pop_debug_group(encoder).unwrap()
                }
                trace::Command::InsertDebugMarker(marker) => self
                    .command_encoder_insert_debug_marker(encoder, &marker)
                    .unwrap(),
                trace::Command::RunComputePass {
                    base,
                    timestamp_writes,
                } => {
                    self.compute_pass_end_with_unresolved_commands(
                        encoder,
                        base,
                        timestamp_writes.as_ref(),
                    )
                    .unwrap();
                }
                trace::Command::RunRenderPass {
                    base,
                    target_colors,
                    target_depth_stencil,
                    timestamp_writes,
                    occlusion_query_set_id,
                } => {
                    self.render_pass_end_with_unresolved_commands(
                        encoder,
                        base,
                        &target_colors,
                        target_depth_stencil.as_ref(),
                        timestamp_writes.as_ref(),
                        occlusion_query_set_id,
                    )
                    .unwrap();
                }
                trace::Command::BuildAccelerationStructuresUnsafeTlas { blas, tlas } => {
                    let blas_iter = blas.iter().map(|x| {
                        let geometries = match &x.geometries {
                            wgc::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                                triangle_geometries,
                            ) => {
                                let iter = triangle_geometries.iter().map(|tg| {
                                    wgc::ray_tracing::BlasTriangleGeometry {
                                        size: &tg.size,
                                        vertex_buffer: tg.vertex_buffer,
                                        index_buffer: tg.index_buffer,
                                        transform_buffer: tg.transform_buffer,
                                        first_vertex: tg.first_vertex,
                                        vertex_stride: tg.vertex_stride,
                                        index_buffer_offset: tg.index_buffer_offset,
                                        transform_buffer_offset: tg.transform_buffer_offset,
                                    }
                                });
                                wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                            }
                        };
                        wgc::ray_tracing::BlasBuildEntry {
                            blas_id: x.blas_id,
                            geometries,
                        }
                    });

                    if !tlas.is_empty() {
                        log::error!("a trace of command_encoder_build_acceleration_structures_unsafe_tlas containing a tlas build is not replayable! skipping tlas build");
                    }

                    self.command_encoder_build_acceleration_structures_unsafe_tlas(
                        encoder,
                        blas_iter,
                        std::iter::empty(),
                    )
                    .unwrap();
                }
                trace::Command::BuildAccelerationStructures { blas, tlas } => {
                    let blas_iter = blas.iter().map(|x| {
                        let geometries = match &x.geometries {
                            wgc::ray_tracing::TraceBlasGeometries::TriangleGeometries(
                                triangle_geometries,
                            ) => {
                                let iter = triangle_geometries.iter().map(|tg| {
                                    wgc::ray_tracing::BlasTriangleGeometry {
                                        size: &tg.size,
                                        vertex_buffer: tg.vertex_buffer,
                                        index_buffer: tg.index_buffer,
                                        transform_buffer: tg.transform_buffer,
                                        first_vertex: tg.first_vertex,
                                        vertex_stride: tg.vertex_stride,
                                        index_buffer_offset: tg.index_buffer_offset,
                                        transform_buffer_offset: tg.transform_buffer_offset,
                                    }
                                });
                                wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                            }
                        };
                        wgc::ray_tracing::BlasBuildEntry {
                            blas_id: x.blas_id,
                            geometries,
                        }
                    });

                    let tlas_iter = tlas.iter().map(|x| {
                        let instances = x.instances.iter().map(|instance| {
                            instance
                                .as_ref()
                                .map(|instance| wgc::ray_tracing::TlasInstance {
                                    blas_id: instance.blas_id,
                                    transform: &instance.transform,
                                    custom_index: instance.custom_index,
                                    mask: instance.mask,
                                })
                        });
                        wgc::ray_tracing::TlasPackage {
                            tlas_id: x.tlas_id,
                            instances: Box::new(instances),
                            lowest_unmodified: x.lowest_unmodified,
                        }
                    });

                    self.command_encoder_build_acceleration_structures(
                        encoder, blas_iter, tlas_iter,
                    )
                    .unwrap();
                }
            }
        }
        let (cmd_buf, error) =
            self.command_encoder_finish(encoder, &wgt::CommandBufferDescriptor { label: None });
        if let Some(e) = error {
            panic!("{e}");
        }
        cmd_buf
    }

    fn process(
        &self,
        device: wgc::id::DeviceId,
        queue: wgc::id::QueueId,
        action: trace::Action,
        dir: &Path,
        comb_manager: &mut wgc::identity::IdentityManager<wgc::id::markers::CommandBuffer>,
    ) {
        use wgc::device::trace::Action;
        log::debug!("action {:?}", action);
        //TODO: find a way to force ID perishing without excessive `maintain()` calls.
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
                let (_, error) = self.device_create_buffer(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::FreeBuffer(id) => {
                self.buffer_destroy(id).unwrap();
            }
            Action::DestroyBuffer(id) => {
                self.buffer_drop(id);
            }
            Action::CreateTexture(id, desc) => {
                let (_, error) = self.device_create_texture(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::FreeTexture(id) => {
                self.texture_destroy(id).unwrap();
            }
            Action::DestroyTexture(id) => {
                self.texture_drop(id);
            }
            Action::CreateTextureView {
                id,
                parent_id,
                desc,
            } => {
                let (_, error) = self.texture_create_view(parent_id, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyTextureView(id) => {
                self.texture_view_drop(id).unwrap();
            }
            Action::CreateSampler(id, desc) => {
                let (_, error) = self.device_create_sampler(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroySampler(id) => {
                self.sampler_drop(id);
            }
            Action::GetSurfaceTexture { id, parent_id } => {
                self.surface_get_current_texture(parent_id, Some(id))
                    .unwrap()
                    .texture_id
                    .unwrap();
            }
            Action::CreateBindGroupLayout(id, desc) => {
                let (_, error) = self.device_create_bind_group_layout(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyBindGroupLayout(id) => {
                self.bind_group_layout_drop(id);
            }
            Action::CreatePipelineLayout(id, desc) => {
                let (_, error) = self.device_create_pipeline_layout(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyPipelineLayout(id) => {
                self.pipeline_layout_drop(id);
            }
            Action::CreateBindGroup(id, desc) => {
                let (_, error) = self.device_create_bind_group(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyBindGroup(id) => {
                self.bind_group_drop(id);
            }
            Action::CreateShaderModule { id, desc, data } => {
                log::debug!("Creating shader from {}", data);
                let code = fs::read_to_string(dir.join(&data)).unwrap();
                let source = if data.ends_with(".wgsl") {
                    wgc::pipeline::ShaderModuleSource::Wgsl(Cow::Owned(code.clone()))
                } else if data.ends_with(".ron") {
                    let module = ron::de::from_str(&code).unwrap();
                    wgc::pipeline::ShaderModuleSource::Naga(module)
                } else {
                    panic!("Unknown shader {data}");
                };
                let (_, error) = self.device_create_shader_module(device, &desc, source, Some(id));
                if let Some(e) = error {
                    println!("shader compilation error:\n---{code}\n---\n{e}");
                }
            }
            Action::DestroyShaderModule(id) => {
                self.shader_module_drop(id);
            }
            Action::CreateComputePipeline {
                id,
                desc,
                implicit_context,
            } => {
                let implicit_ids =
                    implicit_context
                        .as_ref()
                        .map(|ic| wgc::device::ImplicitPipelineIds {
                            root_id: ic.root_id,
                            group_ids: &ic.group_ids,
                        });
                let (_, error) =
                    self.device_create_compute_pipeline(device, &desc, Some(id), implicit_ids);
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyComputePipeline(id) => {
                self.compute_pipeline_drop(id);
            }
            Action::CreateRenderPipeline {
                id,
                desc,
                implicit_context,
            } => {
                let implicit_ids =
                    implicit_context
                        .as_ref()
                        .map(|ic| wgc::device::ImplicitPipelineIds {
                            root_id: ic.root_id,
                            group_ids: &ic.group_ids,
                        });
                let (_, error) =
                    self.device_create_render_pipeline(device, &desc, Some(id), implicit_ids);
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyRenderPipeline(id) => {
                self.render_pipeline_drop(id);
            }
            Action::CreatePipelineCache { id, desc } => {
                let _ = unsafe { self.device_create_pipeline_cache(device, &desc, Some(id)) };
            }
            Action::DestroyPipelineCache(id) => {
                self.pipeline_cache_drop(id);
            }
            Action::CreateRenderBundle { id, desc, base } => {
                let bundle =
                    wgc::command::RenderBundleEncoder::new(&desc, device, Some(base)).unwrap();
                let (_, error) = self.render_bundle_encoder_finish(
                    bundle,
                    &wgt::RenderBundleDescriptor { label: desc.label },
                    Some(id),
                );
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyRenderBundle(id) => {
                self.render_bundle_drop(id);
            }
            Action::CreateQuerySet { id, desc } => {
                let (_, error) = self.device_create_query_set(device, &desc, Some(id));
                if let Some(e) = error {
                    panic!("{e}");
                }
            }
            Action::DestroyQuerySet(id) => {
                self.query_set_drop(id);
            }
            Action::WriteBuffer {
                id,
                data,
                range,
                queued,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                let size = (range.end - range.start) as usize;
                if queued {
                    self.queue_write_buffer(queue, id, range.start, &bin)
                        .unwrap();
                } else {
                    self.device_set_buffer_data(id, range.start, &bin[..size])
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
                self.queue_write_texture(queue, &to, &bin, &layout, &size)
                    .unwrap();
            }
            Action::Submit(_index, ref commands) if commands.is_empty() => {
                self.queue_submit(queue, &[]).unwrap();
            }
            Action::Submit(_index, commands) => {
                let (encoder, error) = self.device_create_command_encoder(
                    device,
                    &wgt::CommandEncoderDescriptor { label: None },
                    Some(comb_manager.process().into_command_encoder_id()),
                );
                if let Some(e) = error {
                    panic!("{e}");
                }
                let cmdbuf = self.encode_commands(encoder, commands);
                self.queue_submit(queue, &[cmdbuf]).unwrap();
            }
            Action::CreateBlas { id, desc, sizes } => {
                self.device_create_blas(device, &desc, sizes, Some(id));
            }
            Action::FreeBlas(id) => {
                self.blas_destroy(id).unwrap();
            }
            Action::DestroyBlas(id) => {
                self.blas_drop(id);
            }
            Action::CreateTlas { id, desc } => {
                self.device_create_tlas(device, &desc, Some(id));
            }
            Action::FreeTlas(id) => {
                self.tlas_destroy(id).unwrap();
            }
            Action::DestroyTlas(id) => {
                self.tlas_drop(id);
            }
        }
    }
}
