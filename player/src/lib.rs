/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! This is a player library for WebGPU traces.
 *
 * # Notes
 * - we call device_maintain_ids() before creating any refcounted resource,
 *   which is basically everything except for BGL and shader modules,
 *   so that we don't accidentally try to use the same ID.
!*/

use wgc::device::trace;

use std::{borrow::Cow, fmt::Debug, fs, marker::PhantomData, path::Path};

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

pub trait GlobalPlay {
    fn encode_commands<B: wgc::hub::GfxBackend>(
        &self,
        encoder: wgc::id::CommandEncoderId,
        commands: Vec<trace::Command>,
    ) -> wgc::id::CommandBufferId;
    fn process<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        action: trace::Action,
        dir: &Path,
        comb_manager: &mut wgc::hub::IdentityManager,
    );
}

impl GlobalPlay for wgc::hub::Global<IdentityPassThroughFactory> {
    fn encode_commands<B: wgc::hub::GfxBackend>(
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
                    .command_encoder_copy_buffer_to_buffer::<B>(
                        encoder, src, src_offset, dst, dst_offset, size,
                    )
                    .unwrap(),
                trace::Command::CopyBufferToTexture { src, dst, size } => self
                    .command_encoder_copy_buffer_to_texture::<B>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToBuffer { src, dst, size } => self
                    .command_encoder_copy_texture_to_buffer::<B>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::CopyTextureToTexture { src, dst, size } => self
                    .command_encoder_copy_texture_to_texture::<B>(encoder, &src, &dst, &size)
                    .unwrap(),
                trace::Command::ClearBuffer { dst, offset, size } => self
                    .command_encoder_clear_buffer::<B>(encoder, dst, offset, size)
                    .unwrap(),
                trace::Command::ClearImage {
                    dst,
                    subresource_range,
                } => self
                    .command_encoder_clear_image::<B>(encoder, dst, subresource_range)
                    .unwrap(),
                trace::Command::WriteTimestamp {
                    query_set_id,
                    query_index,
                } => self
                    .command_encoder_write_timestamp::<B>(encoder, query_set_id, query_index)
                    .unwrap(),
                trace::Command::ResolveQuerySet {
                    query_set_id,
                    start_query,
                    query_count,
                    destination,
                    destination_offset,
                } => self
                    .command_encoder_resolve_query_set::<B>(
                        encoder,
                        query_set_id,
                        start_query,
                        query_count,
                        destination,
                        destination_offset,
                    )
                    .unwrap(),
                trace::Command::RunComputePass { base } => {
                    self.command_encoder_run_compute_pass_impl::<B>(encoder, base.as_ref())
                        .unwrap();
                }
                trace::Command::RunRenderPass {
                    base,
                    target_colors,
                    target_depth_stencil,
                } => {
                    self.command_encoder_run_render_pass_impl::<B>(
                        encoder,
                        base.as_ref(),
                        &target_colors,
                        target_depth_stencil.as_ref(),
                    )
                    .unwrap();
                }
            }
        }
        let (cmd_buf, error) = self
            .command_encoder_finish::<B>(encoder, &wgt::CommandBufferDescriptor { label: None });
        if let Some(e) = error {
            panic!("{:?}", e);
        }
        cmd_buf
    }

    fn process<B: wgc::hub::GfxBackend>(
        &self,
        device: wgc::id::DeviceId,
        action: trace::Action,
        dir: &Path,
        comb_manager: &mut wgc::hub::IdentityManager,
    ) {
        use wgc::device::trace::Action as A;
        log::info!("action {:?}", action);
        //TODO: find a way to force ID perishing without excessive `maintain()` calls.
        match action {
            A::Init { .. } => panic!("Unexpected Action::Init: has to be the first action only"),
            A::CreateSwapChain { .. } | A::PresentSwapChain(_) => {
                panic!("Unexpected SwapChain action: winit feature is not enabled")
            }
            A::CreateBuffer(id, desc) => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.device_create_buffer::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::FreeBuffer(id) => {
                self.buffer_destroy::<B>(id).unwrap();
            }
            A::DestroyBuffer(id) => {
                self.buffer_drop::<B>(id, true);
            }
            A::CreateTexture(id, desc) => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.device_create_texture::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::FreeTexture(id) => {
                self.texture_destroy::<B>(id).unwrap();
            }
            A::DestroyTexture(id) => {
                self.texture_drop::<B>(id, true);
            }
            A::CreateTextureView {
                id,
                parent_id,
                desc,
            } => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.texture_create_view::<B>(parent_id, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyTextureView(id) => {
                self.texture_view_drop::<B>(id, true).unwrap();
            }
            A::CreateSampler(id, desc) => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.device_create_sampler::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroySampler(id) => {
                self.sampler_drop::<B>(id);
            }
            A::GetSwapChainTexture { id, parent_id } => {
                self.device_maintain_ids::<B>(device).unwrap();
                self.swap_chain_get_current_texture_view::<B>(parent_id, id)
                    .unwrap()
                    .view_id
                    .unwrap();
            }
            A::CreateBindGroupLayout(id, desc) => {
                let (_, error) = self.device_create_bind_group_layout::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyBindGroupLayout(id) => {
                self.bind_group_layout_drop::<B>(id);
            }
            A::CreatePipelineLayout(id, desc) => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.device_create_pipeline_layout::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyPipelineLayout(id) => {
                self.pipeline_layout_drop::<B>(id);
            }
            A::CreateBindGroup(id, desc) => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.device_create_bind_group::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyBindGroup(id) => {
                self.bind_group_drop::<B>(id);
            }
            A::CreateShaderModule { id, desc, data } => {
                let source = if data.ends_with(".wgsl") {
                    let code = fs::read_to_string(dir.join(data)).unwrap();
                    wgc::pipeline::ShaderModuleSource::Wgsl(Cow::Owned(code))
                } else {
                    let byte_vec = fs::read(dir.join(&data))
                        .unwrap_or_else(|e| panic!("Unable to open '{}': {:?}", data, e));
                    let spv = byte_vec
                        .chunks(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>();
                    wgc::pipeline::ShaderModuleSource::SpirV(Cow::Owned(spv))
                };
                let (_, error) = self.device_create_shader_module::<B>(device, &desc, source, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyShaderModule(id) => {
                self.shader_module_drop::<B>(id);
            }
            A::CreateComputePipeline {
                id,
                desc,
                implicit_context,
            } => {
                self.device_maintain_ids::<B>(device).unwrap();
                let implicit_ids =
                    implicit_context
                        .as_ref()
                        .map(|ic| wgc::device::ImplicitPipelineIds {
                            root_id: ic.root_id,
                            group_ids: &ic.group_ids,
                        });
                let (_, error) =
                    self.device_create_compute_pipeline::<B>(device, &desc, id, implicit_ids);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyComputePipeline(id) => {
                self.compute_pipeline_drop::<B>(id);
            }
            A::CreateRenderPipeline {
                id,
                desc,
                implicit_context,
            } => {
                self.device_maintain_ids::<B>(device).unwrap();
                let implicit_ids =
                    implicit_context
                        .as_ref()
                        .map(|ic| wgc::device::ImplicitPipelineIds {
                            root_id: ic.root_id,
                            group_ids: &ic.group_ids,
                        });
                let (_, error) =
                    self.device_create_render_pipeline::<B>(device, &desc, id, implicit_ids);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyRenderPipeline(id) => {
                self.render_pipeline_drop::<B>(id);
            }
            A::CreateRenderBundle { id, desc, base } => {
                let bundle =
                    wgc::command::RenderBundleEncoder::new(&desc, device, Some(base)).unwrap();
                let (_, error) = self.render_bundle_encoder_finish::<B>(
                    bundle,
                    &wgt::RenderBundleDescriptor { label: desc.label },
                    id,
                );
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyRenderBundle(id) => {
                self.render_bundle_drop::<B>(id);
            }
            A::CreateQuerySet { id, desc } => {
                self.device_maintain_ids::<B>(device).unwrap();
                let (_, error) = self.device_create_query_set::<B>(device, &desc, id);
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
            }
            A::DestroyQuerySet(id) => {
                self.query_set_drop::<B>(id);
            }
            A::WriteBuffer {
                id,
                data,
                range,
                queued,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                let size = (range.end - range.start) as usize;
                if queued {
                    self.queue_write_buffer::<B>(device, id, range.start, &bin)
                        .unwrap();
                } else {
                    self.device_wait_for_buffer::<B>(device, id).unwrap();
                    self.device_set_buffer_sub_data::<B>(device, id, range.start, &bin[..size])
                        .unwrap();
                }
            }
            A::WriteTexture {
                to,
                data,
                layout,
                size,
            } => {
                let bin = std::fs::read(dir.join(data)).unwrap();
                self.queue_write_texture::<B>(device, &to, &bin, &layout, &size)
                    .unwrap();
            }
            A::Submit(_index, ref commands) if commands.is_empty() => {
                self.queue_submit::<B>(device, &[]).unwrap();
            }
            A::Submit(_index, commands) => {
                let (encoder, error) = self.device_create_command_encoder::<B>(
                    device,
                    &wgt::CommandEncoderDescriptor { label: None },
                    comb_manager.alloc(device.backend()),
                );
                if let Some(e) = error {
                    panic!("{:?}", e);
                }
                let cmdbuf = self.encode_commands::<B>(encoder, commands);
                self.queue_submit::<B>(device, &[cmdbuf]).unwrap();
            }
        }
    }
}
