/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    command::{RawPass, RenderCommand},
    conv,
    device::RenderPassContext,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Input, Token},
    id,
    resource::BufferUse,
    track::TrackerSet,
    LifeGuard, RefCount,
};
use arrayvec::ArrayVec;
use peek_poke::Peek;

#[derive(Debug)]
pub struct RenderBundleEncoderDescriptor<'a> {
    pub color_formats: &'a [wgt::TextureFormat],
    pub depth_stencil_format: Option<wgt::TextureFormat>,
    pub sample_count: u32,
}

#[derive(Debug)]
pub struct RenderBundleEncoder {
    pub(crate) raw: RawPass<id::DeviceId>,
    pub(crate) context: RenderPassContext,
    pub(crate) sample_count: u8,
    pub(crate) backend: wgt::Backend,
}

impl RenderBundleEncoder {
    pub fn new(
        desc: &RenderBundleEncoderDescriptor,
        device_id: id::DeviceId,
        backend: wgt::Backend,
    ) -> Self {
        RenderBundleEncoder {
            raw: RawPass::from_vec::<RenderCommand>(Vec::with_capacity(1), device_id),
            context: RenderPassContext {
                colors: desc.color_formats.iter().cloned().collect(),
                resolves: ArrayVec::new(),
                depth_stencil: desc.depth_stencil_format,
            },
            sample_count: {
                let sc = desc.sample_count;
                assert!(sc == 0 || sc > 32 || !conv::is_power_of_two(sc));
                sc as u8
            },
            backend,
        }
    }

    pub fn destroy(mut self) {
        unsafe { self.raw.invalidate() };
    }
}

//Note: here, `RenderBundle` is just wrapping a raw stream of render commands.
// The plan is to back it by an actual Vulkan secondary buffer, D3D12 Bundle,
// or Metal indirect command buffer.
//Note: there is no API tracing support for `RenderBundle` yet.
// It's transparent with regards to the submitted render passes.
#[derive(Debug)]
pub struct RenderBundle {
    pub(crate) device_ref_count: RefCount,
    pub(crate) raw: RawPass<id::DeviceId>,
    pub(crate) trackers: TrackerSet,
    pub(crate) context: RenderPassContext,
    pub(crate) sample_count: u8,
    pub(crate) life_guard: LifeGuard,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn render_bundle_encoder_finish<B: GfxBackend>(
        &self,
        bundle_encoder: RenderBundleEncoder,
        _desc: &wgt::RenderBundleDescriptor,
        id_in: Input<G, id::RenderBundleId>,
    ) -> id::RenderBundleId {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let render_bundle = {
            let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
            let (pipeline_guard, mut token) = hub.render_pipelines.read(&mut token);
            let (buffer_guard, _) = hub.buffers.read(&mut token);

            let mut trackers = TrackerSet::new(bundle_encoder.backend);

            // populate the trackers and validate the commands
            #[allow(trivial_casts)] // erroneous warning!
            let mut peeker = bundle_encoder.raw.base as *const u8;
            #[allow(trivial_casts)] // erroneous warning!
            let raw_data_end = bundle_encoder.raw.data as *const _;
            let mut command = RenderCommand::Draw {
                vertex_count: 0,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            };
            while peeker != raw_data_end {
                peeker = unsafe { RenderCommand::peek_from(peeker, &mut command) };
                //TODO: find a safer way to enforce this without the `End` command
                assert!(peeker <= raw_data_end);
                match command {
                    RenderCommand::SetBindGroup {
                        index: _,
                        num_dynamic_offsets,
                        bind_group_id,
                        phantom_offsets,
                    } => {
                        let (new_peeker, offsets) = unsafe {
                            phantom_offsets.decode_unaligned(
                                peeker,
                                num_dynamic_offsets as usize,
                                raw_data_end,
                            )
                        };
                        peeker = new_peeker;

                        if cfg!(debug_assertions) {
                            for off in offsets {
                                assert_eq!(
                                    *off as wgt::BufferAddress % wgt::BIND_BUFFER_ALIGNMENT,
                                    0,
                                    "Misaligned dynamic buffer offset: {} does not align with {}",
                                    off,
                                    wgt::BIND_BUFFER_ALIGNMENT
                                );
                            }
                        }

                        let bind_group = trackers
                            .bind_groups
                            .use_extend(&*bind_group_guard, bind_group_id, (), ())
                            .unwrap();
                        assert_eq!(bind_group.dynamic_count, offsets.len());

                        trackers.merge_extend(&bind_group.used);
                    }
                    RenderCommand::SetPipeline(pipeline_id) => {
                        let pipeline = trackers
                            .render_pipes
                            .use_extend(&*pipeline_guard, pipeline_id, (), ())
                            .unwrap();

                        assert!(
                            bundle_encoder.context.compatible(&pipeline.pass_context),
                            "The render pipeline output formats do not match render pass attachment formats!"
                        );
                        assert_eq!(
                            pipeline.sample_count, bundle_encoder.sample_count,
                            "The render pipeline and renderpass have mismatching sample_count"
                        );
                        //TODO: check read-only depth
                    }
                    RenderCommand::SetIndexBuffer { buffer_id, .. } => {
                        let buffer = trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDEX)
                            .unwrap();
                        assert!(buffer.usage.contains(wgt::BufferUsage::INDEX), "An invalid setIndexBuffer call has been made. The buffer usage is {:?} which does not contain required usage INDEX", buffer.usage);
                    }
                    RenderCommand::SetVertexBuffer { buffer_id, .. } => {
                        let buffer = trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::VERTEX)
                            .unwrap();
                        assert!(buffer.usage.contains(wgt::BufferUsage::VERTEX), "An invalid setVertexBuffer call has been made. The buffer usage is {:?} which does not contain required usage VERTEX", buffer.usage);
                    }
                    RenderCommand::Draw { .. } | RenderCommand::DrawIndexed { .. } => {}
                    RenderCommand::DrawIndirect {
                        buffer_id,
                        offset: _,
                    }
                    | RenderCommand::DrawIndexedIndirect {
                        buffer_id,
                        offset: _,
                    } => {
                        let buffer = trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                            .unwrap();
                        assert!(buffer.usage.contains(wgt::BufferUsage::INDIRECT), "An invalid draw(Indexed)Indirect call has been made. The buffer usage is {:?} which does not contain required usage INDIRECT", buffer.usage);
                    }
                    RenderCommand::SetBlendColor(_)
                    | RenderCommand::SetStencilReference(_)
                    | RenderCommand::SetViewport { .. }
                    | RenderCommand::SetScissor(_)
                    | RenderCommand::End => unreachable!("not support by a render bundle"),
                }
            }

            log::debug!("Render bundle {:#?}", trackers);
            //TODO: check if the device is still alive
            let device = &device_guard[bundle_encoder.raw.parent];
            RenderBundle {
                device_ref_count: device.life_guard.add_ref(),
                raw: bundle_encoder.raw,
                trackers,
                context: bundle_encoder.context,
                sample_count: bundle_encoder.sample_count,
                life_guard: LifeGuard::new(),
            }
        };

        hub.render_bundles
            .register_identity(id_in, render_bundle, &mut token)
    }
}

pub mod bundle_ffi {
    use super::{super::PhantomSlice, RenderBundleEncoder, RenderCommand};
    use crate::{id, RawString};
    use std::{convert::TryInto, slice};
    use wgt::{BufferAddress, BufferSize, DynamicOffset};

    /// # Safety
    ///
    /// This function is unsafe as there is no guarantee that the given pointer is
    /// valid for `offset_length` elements.
    // TODO: There might be other safety issues, such as using the unsafe
    // `RawPass::encode` and `RawPass::encode_slice`.
    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_bind_group(
        bundle_encoder: &mut RenderBundleEncoder,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        bundle_encoder.raw.encode(&RenderCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
            phantom_offsets: PhantomSlice::default(),
        });
        bundle_encoder
            .raw
            .encode_slice(slice::from_raw_parts(offsets, offset_length));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_pipeline(
        bundle_encoder: &mut RenderBundleEncoder,
        pipeline_id: id::RenderPipelineId,
    ) {
        bundle_encoder
            .raw
            .encode(&RenderCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_index_buffer(
        bundle_encoder: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: BufferSize,
    ) {
        bundle_encoder.raw.encode(&RenderCommand::SetIndexBuffer {
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_set_vertex_buffer(
        bundle_encoder: &mut RenderBundleEncoder,
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: BufferSize,
    ) {
        bundle_encoder.raw.encode(&RenderCommand::SetVertexBuffer {
            slot,
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_draw(
        bundle_encoder: &mut RenderBundleEncoder,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        bundle_encoder.raw.encode(&RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_draw_indexed(
        bundle_encoder: &mut RenderBundleEncoder,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        bundle_encoder.raw.encode(&RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_draw_indirect(
        bundle_encoder: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        bundle_encoder
            .raw
            .encode(&RenderCommand::DrawIndirect { buffer_id, offset });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_pass_bundle_indexed_indirect(
        bundle_encoder: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        bundle_encoder
            .raw
            .encode(&RenderCommand::DrawIndexedIndirect { buffer_id, offset });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_push_debug_group(
        _bundle_encoder: &mut RenderBundleEncoder,
        _label: RawString,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_pop_debug_group(
        _bundle_encoder: &mut RenderBundleEncoder,
    ) {
        //TODO
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_insert_debug_marker(
        _bundle_encoder: &mut RenderBundleEncoder,
        _label: RawString,
    ) {
        //TODO
    }
}
