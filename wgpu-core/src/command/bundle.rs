/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! Render Bundles

    ## Software implementation

    The path from nothing to using a render bundle consists of 3 phases.

    ### Initial command encoding

    User creates a `RenderBundleEncoder` and populates it by issuing commands
    from `bundle_ffi` module, just like with `RenderPass`, except that the
    set of available commands is reduced. Everything is written into a `RawPass`.

    ### Bundle baking

    Once the commands are encoded, user calls `render_bundle_encoder_finish`.
    This is perhaps the most complex part of the logic. It consumes the
    commands stored in `RawPass`, while validating everything, tracking the state,
    and re-recording the commands into a separate `Vec<RenderCommand>`. It
    doesn't actually execute any commands.

    What's more important, is that the produced vector of commands is "normalized",
    which means it can be executed verbatim without any state tracking. More
    formally, "normalized" command stream guarantees that any state required by
    a draw call is set explicitly by one of the commands between the draw call
    and the last changing of the pipeline.

    ### Execution

    When the bundle is used in an actual render pass, `RenderBundle::execute` is
    called. It goes through the commands and issues them into the native command
    buffer. Thanks to the "normalized" property, it doesn't track any bind group
    invalidations or index format changes.
!*/

use crate::{
    command::{BasePass, RenderCommand},
    conv,
    device::{AttachmentData, Label, RenderPassContext, MAX_VERTEX_BUFFERS},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Input, Storage, Token},
    id,
    resource::BufferUse,
    span,
    track::TrackerSet,
    LifeGuard, RefCount, Stored, MAX_BIND_GROUPS,
};
use arrayvec::ArrayVec;
use std::{borrow::Borrow, iter, marker::PhantomData, ops::Range};

#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub struct RenderBundleEncoder {
    base: BasePass<RenderCommand>,
    parent_id: id::DeviceId,
    pub(crate) context: RenderPassContext,
}

impl RenderBundleEncoder {
    pub fn new(
        desc: &wgt::RenderBundleEncoderDescriptor,
        parent_id: id::DeviceId,
        base: Option<BasePass<RenderCommand>>,
    ) -> Self {
        span!(_guard, TRACE, "RenderBundleEncoder::new");
        RenderBundleEncoder {
            base: base.unwrap_or_else(BasePass::new),
            parent_id,
            context: RenderPassContext {
                attachments: AttachmentData {
                    colors: desc.color_formats.iter().cloned().collect(),
                    resolves: ArrayVec::new(),
                    depth_stencil: desc.depth_stencil_format,
                },
                sample_count: {
                    let sc = desc.sample_count;
                    assert!(sc != 0 && sc <= 32 && conv::is_power_of_two(sc));
                    sc as u8
                },
            },
        }
    }

    pub fn parent(&self) -> id::DeviceId {
        self.parent_id
    }
}

//Note: here, `RenderBundle` is just wrapping a raw stream of render commands.
// The plan is to back it by an actual Vulkan secondary buffer, D3D12 Bundle,
// or Metal indirect command buffer.
#[derive(Debug)]
pub struct RenderBundle {
    // Normalized command stream. It can be executed verbatim,
    // without re-binding anything on the pipeline change.
    base: BasePass<RenderCommand>,
    pub(crate) device_id: Stored<id::DeviceId>,
    pub(crate) used: TrackerSet,
    pub(crate) context: RenderPassContext,
    pub(crate) life_guard: LifeGuard,
}

unsafe impl Send for RenderBundle {}
unsafe impl Sync for RenderBundle {}

impl RenderBundle {
    /// Actually encode the contents into a native command buffer.
    ///
    /// This is partially duplicating the logic of `command_encoder_run_render_pass`.
    /// However the point of this function is to be lighter, since we already had
    /// a chance to go through the commands in `render_bundle_encoder_finish`.
    pub(crate) unsafe fn execute<B: GfxBackend>(
        &self,
        comb: &mut B::CommandBuffer,
        pipeline_layout_guard: &Storage<
            crate::binding_model::PipelineLayout<B>,
            id::PipelineLayoutId,
        >,
        bind_group_guard: &Storage<crate::binding_model::BindGroup<B>, id::BindGroupId>,
        pipeline_guard: &Storage<crate::pipeline::RenderPipeline<B>, id::RenderPipelineId>,
        buffer_guard: &Storage<crate::resource::Buffer<B>, id::BufferId>,
    ) {
        use hal::command::CommandBuffer as _;

        let mut offsets = self.base.dynamic_offsets.as_slice();
        let mut index_type = hal::IndexType::U16;
        let mut pipeline_layout_id = None::<id::PipelineLayoutId>;

        for command in self.base.commands.iter() {
            match *command {
                RenderCommand::SetBindGroup {
                    index,
                    num_dynamic_offsets,
                    bind_group_id,
                } => {
                    let bind_group = &bind_group_guard[bind_group_id];
                    comb.bind_graphics_descriptor_sets(
                        &pipeline_layout_guard[pipeline_layout_id.unwrap()].raw,
                        index as usize,
                        iter::once(bind_group.raw.raw()),
                        &offsets[..num_dynamic_offsets as usize],
                    );
                    offsets = &offsets[num_dynamic_offsets as usize..];
                }
                RenderCommand::SetPipeline(pipeline_id) => {
                    let pipeline = &pipeline_guard[pipeline_id];
                    comb.bind_graphics_pipeline(&pipeline.raw);
                    index_type = conv::map_index_format(pipeline.index_format);
                    pipeline_layout_id = Some(pipeline.layout_id.value);
                }
                RenderCommand::SetIndexBuffer {
                    buffer_id,
                    offset,
                    size,
                } => {
                    let buffer = &buffer_guard[buffer_id];
                    let view = hal::buffer::IndexBufferView {
                        buffer: &buffer.raw,
                        range: hal::buffer::SubRange {
                            offset,
                            size: size.map(|s| s.get()),
                        },
                        index_type,
                    };

                    comb.bind_index_buffer(view);
                }
                RenderCommand::SetVertexBuffer {
                    slot,
                    buffer_id,
                    offset,
                    size,
                } => {
                    let buffer = &buffer_guard[buffer_id];
                    let range = hal::buffer::SubRange {
                        offset,
                        size: size.map(|s| s.get()),
                    };
                    comb.bind_vertex_buffers(slot, iter::once((&buffer.raw, range)));
                }
                RenderCommand::Draw {
                    vertex_count,
                    instance_count,
                    first_vertex,
                    first_instance,
                } => {
                    comb.draw(
                        first_vertex..first_vertex + vertex_count,
                        first_instance..first_instance + instance_count,
                    );
                }
                RenderCommand::DrawIndexed {
                    index_count,
                    instance_count,
                    first_index,
                    base_vertex,
                    first_instance,
                } => {
                    comb.draw_indexed(
                        first_index..first_index + index_count,
                        base_vertex,
                        first_instance..first_instance + instance_count,
                    );
                }
                RenderCommand::DrawIndirect { buffer_id, offset } => {
                    let buffer = &buffer_guard[buffer_id];
                    comb.draw_indirect(&buffer.raw, offset, 1, 0);
                }
                RenderCommand::DrawIndexedIndirect { buffer_id, offset } => {
                    let buffer = &buffer_guard[buffer_id];
                    comb.draw_indexed_indirect(&buffer.raw, offset, 1, 0);
                }
                RenderCommand::PushDebugGroup { color: _, len: _ } => unimplemented!(),
                RenderCommand::InsertDebugMarker { color: _, len: _ } => unimplemented!(),
                RenderCommand::PopDebugGroup => unimplemented!(),
                RenderCommand::ExecuteBundle(_)
                | RenderCommand::SetBlendColor(_)
                | RenderCommand::SetStencilReference(_)
                | RenderCommand::SetViewport { .. }
                | RenderCommand::SetScissor(_) => unreachable!(),
            }
        }
    }
}

impl Borrow<RefCount> for RenderBundle {
    fn borrow(&self) -> &RefCount {
        self.life_guard.ref_count.as_ref().unwrap()
    }
}

#[derive(Debug)]
struct IndexState {
    buffer: Option<id::BufferId>,
    format: wgt::IndexFormat,
    range: Range<wgt::BufferAddress>,
    is_dirty: bool,
}

impl IndexState {
    fn new() -> Self {
        IndexState {
            buffer: None,
            format: wgt::IndexFormat::default(),
            range: 0..0,
            is_dirty: false,
        }
    }

    fn limit(&self) -> u32 {
        assert!(self.buffer.is_some());
        let bytes_per_index = match self.format {
            wgt::IndexFormat::Uint16 => 2,
            wgt::IndexFormat::Uint32 => 4,
        };
        ((self.range.end - self.range.start) / bytes_per_index) as u32
    }

    fn flush(&mut self) -> Option<RenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(RenderCommand::SetIndexBuffer {
                buffer_id: self.buffer.unwrap(),
                offset: self.range.start,
                size: wgt::BufferSize::new(self.range.end - self.range.start),
            })
        } else {
            None
        }
    }

    fn set_format(&mut self, format: wgt::IndexFormat) {
        if self.format != format {
            self.format = format;
            self.is_dirty = true;
        }
    }

    fn set_buffer(&mut self, id: id::BufferId, range: Range<wgt::BufferAddress>) {
        self.buffer = Some(id);
        self.range = range;
        self.is_dirty = true;
    }
}

#[derive(Debug)]
struct VertexState {
    buffer: Option<id::BufferId>,
    range: Range<wgt::BufferAddress>,
    stride: wgt::BufferAddress,
    rate: wgt::InputStepMode,
    is_dirty: bool,
}

impl VertexState {
    fn new() -> Self {
        VertexState {
            buffer: None,
            range: 0..0,
            stride: 0,
            rate: wgt::InputStepMode::Vertex,
            is_dirty: false,
        }
    }

    fn set_buffer(&mut self, buffer_id: id::BufferId, range: Range<wgt::BufferAddress>) {
        self.buffer = Some(buffer_id);
        self.range = range;
        self.is_dirty = true;
    }

    fn flush(&mut self, slot: u32) -> Option<RenderCommand> {
        if self.is_dirty {
            self.is_dirty = false;
            Some(RenderCommand::SetVertexBuffer {
                slot,
                buffer_id: self.buffer.unwrap(),
                offset: self.range.start,
                size: wgt::BufferSize::new(self.range.end - self.range.start),
            })
        } else {
            None
        }
    }
}

#[derive(Debug)]
struct BindState {
    bind_group: Option<(id::BindGroupId, id::BindGroupLayoutId)>,
    dynamic_offsets: Range<usize>,
    is_dirty: bool,
}

impl BindState {
    fn new() -> Self {
        BindState {
            bind_group: None,
            dynamic_offsets: 0..0,
            is_dirty: false,
        }
    }

    fn set_group(
        &mut self,
        bind_group_id: id::BindGroupId,
        layout_id: id::BindGroupLayoutId,
        dyn_offset: usize,
        dyn_count: usize,
    ) -> bool {
        match self.bind_group {
            Some((bg_id, _)) if bg_id == bind_group_id && dyn_count == 0 => false,
            _ => {
                self.bind_group = Some((bind_group_id, layout_id));
                self.dynamic_offsets = dyn_offset..dyn_offset + dyn_count;
                self.is_dirty = true;
                true
            }
        }
    }
}

#[derive(Debug)]
struct State {
    trackers: TrackerSet,
    index: IndexState,
    vertex: ArrayVec<[VertexState; MAX_VERTEX_BUFFERS]>,
    bind: ArrayVec<[BindState; MAX_BIND_GROUPS]>,
    raw_dynamic_offsets: Vec<wgt::DynamicOffset>,
    flat_dynamic_offsets: Vec<wgt::DynamicOffset>,
    used_bind_groups: usize,
}

impl State {
    fn vertex_limits(&self) -> (u32, u32) {
        let mut vertex_limit = !0;
        let mut instance_limit = !0;
        for vbs in &self.vertex {
            if vbs.stride == 0 {
                continue;
            }
            let limit = ((vbs.range.end - vbs.range.start) / vbs.stride) as u32;
            match vbs.rate {
                wgt::InputStepMode::Vertex => vertex_limit = vertex_limit.min(limit),
                wgt::InputStepMode::Instance => instance_limit = instance_limit.min(limit),
            }
        }
        (vertex_limit, instance_limit)
    }

    fn invalidate_group_from(&mut self, slot: usize) {
        for bind in self.bind[slot..].iter_mut() {
            if bind.bind_group.is_some() {
                bind.is_dirty = true;
            }
        }
    }

    fn set_bind_group(
        &mut self,
        slot: u8,
        bind_group_id: id::BindGroupId,
        layout_id: id::BindGroupLayoutId,
        offsets: &[wgt::DynamicOffset],
    ) {
        if self.bind[slot as usize].set_group(
            bind_group_id,
            layout_id,
            self.raw_dynamic_offsets.len(),
            offsets.len(),
        ) {
            self.invalidate_group_from(slot as usize + 1);
        }
        self.raw_dynamic_offsets.extend(offsets);
    }

    fn set_pipeline(
        &mut self,
        index_format: wgt::IndexFormat,
        vertex_strides: &[(wgt::BufferAddress, wgt::InputStepMode)],
        layout_ids: &[Stored<id::BindGroupLayoutId>],
    ) {
        self.index.set_format(index_format);
        for (vs, &(stride, step_mode)) in self.vertex.iter_mut().zip(vertex_strides) {
            if vs.stride != stride || vs.rate != step_mode {
                vs.stride = stride;
                vs.rate = step_mode;
                vs.is_dirty = true;
            }
        }
        self.used_bind_groups = layout_ids.len();
        let invalid_from = self
            .bind
            .iter()
            .zip(layout_ids)
            .position(|(bs, layout_id)| match bs.bind_group {
                Some((_, bgl_id)) => bgl_id != layout_id.value,
                None => false,
            });
        if let Some(slot) = invalid_from {
            self.invalidate_group_from(slot);
        }
    }

    fn flush_vertices(&mut self) -> impl Iterator<Item = RenderCommand> + '_ {
        self.vertex
            .iter_mut()
            .enumerate()
            .flat_map(|(i, vs)| vs.flush(i as u32))
    }

    fn flush_binds(&mut self) -> impl Iterator<Item = RenderCommand> + '_ {
        for bs in self.bind[..self.used_bind_groups].iter() {
            if bs.is_dirty {
                self.flat_dynamic_offsets
                    .extend_from_slice(&self.raw_dynamic_offsets[bs.dynamic_offsets.clone()]);
            }
        }
        self.bind
            .iter_mut()
            .take(self.used_bind_groups)
            .enumerate()
            .flat_map(|(i, bs)| {
                if bs.is_dirty {
                    bs.is_dirty = false;
                    Some(RenderCommand::SetBindGroup {
                        index: i as u8,
                        bind_group_id: bs.bind_group.unwrap().0,
                        num_dynamic_offsets: (bs.dynamic_offsets.end - bs.dynamic_offsets.start)
                            as u8,
                    })
                } else {
                    None
                }
            })
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn render_bundle_encoder_finish<B: GfxBackend>(
        &self,
        bundle_encoder: RenderBundleEncoder,
        desc: &wgt::RenderBundleDescriptor<Label>,
        id_in: Input<G, id::RenderBundleId>,
    ) -> id::RenderBundleId {
        span!(_guard, TRACE, "RenderBundleEncoder::finish");
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let device = &device_guard[bundle_encoder.parent_id];
        let render_bundle = {
            let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
            let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
            let (pipeline_guard, mut token) = hub.render_pipelines.read(&mut token);
            let (buffer_guard, _) = hub.buffers.read(&mut token);

            let mut state = State {
                trackers: TrackerSet::new(bundle_encoder.parent_id.backend()),
                index: IndexState::new(),
                vertex: (0..MAX_VERTEX_BUFFERS)
                    .map(|_| VertexState::new())
                    .collect(),
                bind: (0..MAX_BIND_GROUPS).map(|_| BindState::new()).collect(),
                raw_dynamic_offsets: Vec::new(),
                flat_dynamic_offsets: Vec::new(),
                used_bind_groups: 0,
            };
            let mut commands = Vec::new();
            let mut base = bundle_encoder.base.as_ref();

            for &command in base.commands {
                match command {
                    RenderCommand::SetBindGroup {
                        index,
                        num_dynamic_offsets,
                        bind_group_id,
                    } => {
                        let offsets = &base.dynamic_offsets[..num_dynamic_offsets as usize];
                        base.dynamic_offsets =
                            &base.dynamic_offsets[num_dynamic_offsets as usize..];
                        for off in offsets {
                            assert_eq!(
                                *off as wgt::BufferAddress % wgt::BIND_BUFFER_ALIGNMENT,
                                0,
                                "Misaligned dynamic buffer offset: {} does not align with {}",
                                off,
                                wgt::BIND_BUFFER_ALIGNMENT
                            );
                        }

                        let bind_group = state
                            .trackers
                            .bind_groups
                            .use_extend(&*bind_group_guard, bind_group_id, (), ())
                            .unwrap();
                        assert_eq!(bind_group.dynamic_count, offsets.len());

                        state.set_bind_group(index, bind_group_id, bind_group.layout_id, offsets);
                        state.trackers.merge_extend(&bind_group.used);
                    }
                    RenderCommand::SetPipeline(pipeline_id) => {
                        let pipeline = state
                            .trackers
                            .render_pipes
                            .use_extend(&*pipeline_guard, pipeline_id, (), ())
                            .unwrap();

                        assert!(
                            bundle_encoder.context.compatible(&pipeline.pass_context),
                            "The render pipeline output formats and sample counts do not match render pass attachment formats!"
                        );
                        //TODO: check read-only depth

                        let layout = &pipeline_layout_guard[pipeline.layout_id.value];

                        state.set_pipeline(
                            pipeline.index_format,
                            &pipeline.vertex_strides,
                            &layout.bind_group_layout_ids,
                        );
                        commands.push(command);
                    }
                    RenderCommand::SetIndexBuffer {
                        buffer_id,
                        offset,
                        size,
                    } => {
                        let buffer = state
                            .trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDEX)
                            .unwrap();
                        assert!(buffer.usage.contains(wgt::BufferUsage::INDEX), "An invalid setIndexBuffer call has been made. The buffer usage is {:?} which does not contain required usage INDEX", buffer.usage);

                        let end = match size {
                            Some(s) => offset + s.get(),
                            None => buffer.size,
                        };
                        state.index.set_buffer(buffer_id, offset..end);
                    }
                    RenderCommand::SetVertexBuffer {
                        slot,
                        buffer_id,
                        offset,
                        size,
                    } => {
                        let buffer = state
                            .trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::VERTEX)
                            .unwrap();
                        assert!(
                            buffer.usage.contains(wgt::BufferUsage::VERTEX),
                            "An invalid setVertexBuffer call has been made. The buffer usage is {:?} which does not contain required usage VERTEX",
                            buffer.usage
                        );

                        let end = match size {
                            Some(s) => offset + s.get(),
                            None => buffer.size,
                        };
                        state.vertex[slot as usize].set_buffer(buffer_id, offset..end);
                    }
                    RenderCommand::Draw {
                        vertex_count,
                        instance_count,
                        first_vertex,
                        first_instance,
                    } => {
                        let (vertex_limit, instance_limit) = state.vertex_limits();
                        assert!(
                            first_vertex + vertex_count <= vertex_limit,
                            "Vertex {} extends beyond limit {}",
                            first_vertex + vertex_count,
                            vertex_limit
                        );
                        assert!(
                            first_instance + instance_count <= instance_limit,
                            "Instance {} extends beyond limit {}",
                            first_instance + instance_count,
                            instance_limit
                        );
                        commands.extend(state.flush_vertices());
                        commands.extend(state.flush_binds());
                        commands.push(command);
                    }
                    RenderCommand::DrawIndexed {
                        index_count,
                        instance_count,
                        first_index,
                        base_vertex: _,
                        first_instance,
                    } => {
                        //TODO: validate that base_vertex + max_index() is within the provided range
                        let (_, instance_limit) = state.vertex_limits();
                        let index_limit = state.index.limit();
                        assert!(
                            first_index + index_count <= index_limit,
                            "Index {} extends beyond limit {}",
                            first_index + index_count,
                            index_limit
                        );
                        assert!(
                            first_instance + instance_count <= instance_limit,
                            "Instance {} extends beyond limit {}",
                            first_instance + instance_count,
                            instance_limit
                        );
                        commands.extend(state.index.flush());
                        commands.extend(state.flush_vertices());
                        commands.extend(state.flush_binds());
                        commands.push(command);
                    }
                    RenderCommand::DrawIndirect {
                        buffer_id,
                        offset: _,
                    } => {
                        let buffer = state
                            .trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                            .unwrap();
                        assert!(
                            buffer.usage.contains(wgt::BufferUsage::INDIRECT),
                            "An invalid drawIndirect call has been made. The buffer usage is {:?} which does not contain required usage INDIRECT",
                            buffer.usage
                        );

                        commands.extend(state.flush_vertices());
                        commands.extend(state.flush_binds());
                        commands.push(command);
                    }
                    RenderCommand::DrawIndexedIndirect {
                        buffer_id,
                        offset: _,
                    } => {
                        let buffer = state
                            .trackers
                            .buffers
                            .use_extend(&*buffer_guard, buffer_id, (), BufferUse::INDIRECT)
                            .unwrap();
                        assert!(
                            buffer.usage.contains(wgt::BufferUsage::INDIRECT),
                            "An invalid drawIndexedIndirect call has been made. The buffer usage is {:?} which does not contain required usage INDIRECT",
                            buffer.usage
                        );

                        commands.extend(state.index.flush());
                        commands.extend(state.flush_vertices());
                        commands.extend(state.flush_binds());
                        commands.push(command);
                    }
                    RenderCommand::PushDebugGroup { color: _, len: _ } => unimplemented!(),
                    RenderCommand::InsertDebugMarker { color: _, len: _ } => unimplemented!(),
                    RenderCommand::PopDebugGroup => unimplemented!(),
                    RenderCommand::ExecuteBundle(_)
                    | RenderCommand::SetBlendColor(_)
                    | RenderCommand::SetStencilReference(_)
                    | RenderCommand::SetViewport { .. }
                    | RenderCommand::SetScissor(_) => {
                        unreachable!("not supported by a render bundle")
                    }
                }
            }

            log::debug!("Render bundle {:?} = {:#?}", id_in, state.trackers);
            let _ = desc.label; //TODO: actually use
                                //TODO: check if the device is still alive
            RenderBundle {
                base: BasePass {
                    commands,
                    dynamic_offsets: state.flat_dynamic_offsets,
                    string_data: Vec::new(),
                },
                device_id: Stored {
                    value: bundle_encoder.parent_id,
                    ref_count: device.life_guard.add_ref(),
                },
                used: state.trackers,
                context: bundle_encoder.context,
                life_guard: LifeGuard::new(),
            }
        };

        let ref_count = render_bundle.life_guard.add_ref();
        let id = hub
            .render_bundles
            .register_identity(id_in, render_bundle, &mut token);

        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => {
                use crate::device::trace;
                let (bundle_guard, _) = hub.render_bundles.read(&mut token);
                let bundle = &bundle_guard[id];
                trace.lock().add(trace::Action::CreateRenderBundle {
                    id,
                    desc: trace::RenderBundleDescriptor::new(desc.label, &bundle.context),
                    base: BasePass::from_ref(bundle.base.as_ref()),
                });
            }
            None => {}
        }

        device
            .trackers
            .lock()
            .bundles
            .init(id, ref_count, PhantomData)
            .unwrap();
        id
    }
}

pub mod bundle_ffi {
    use super::{RenderBundleEncoder, RenderCommand};
    use crate::{id, RawString, span};
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
        bundle: &mut RenderBundleEncoder,
        index: u32,
        bind_group_id: id::BindGroupId,
        offsets: *const DynamicOffset,
        offset_length: usize,
    ) {
        span!(_guard, TRACE, "RenderBundle::set_bind_group");
        bundle.base.commands.push(RenderCommand::SetBindGroup {
            index: index.try_into().unwrap(),
            num_dynamic_offsets: offset_length.try_into().unwrap(),
            bind_group_id,
        });
        bundle
            .base
            .dynamic_offsets
            .extend_from_slice(slice::from_raw_parts(offsets, offset_length));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_pipeline(
        bundle: &mut RenderBundleEncoder,
        pipeline_id: id::RenderPipelineId,
    ) {
        span!(_guard, TRACE, "RenderBundle::set_pipeline");
        bundle
            .base
            .commands
            .push(RenderCommand::SetPipeline(pipeline_id));
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_index_buffer(
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        span!(_guard, TRACE, "RenderBundle::set_index_buffer");
        bundle.base.commands.push(RenderCommand::SetIndexBuffer {
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_set_vertex_buffer(
        bundle: &mut RenderBundleEncoder,
        slot: u32,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) {
        span!(_guard, TRACE, "RenderBundle::set_vertex_buffer");
        bundle.base.commands.push(RenderCommand::SetVertexBuffer {
            slot,
            buffer_id,
            offset,
            size,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_draw(
        bundle: &mut RenderBundleEncoder,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        span!(_guard, TRACE, "RenderBundle::draw");
        bundle.base.commands.push(RenderCommand::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_draw_indexed(
        bundle: &mut RenderBundleEncoder,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        span!(_guard, TRACE, "RenderBundle::draw_indexed");
        bundle.base.commands.push(RenderCommand::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_bundle_draw_indirect(
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        span!(_guard, TRACE, "RenderBundle::draw_indirect");
        bundle
            .base
            .commands
            .push(RenderCommand::DrawIndirect { buffer_id, offset });
    }

    #[no_mangle]
    pub extern "C" fn wgpu_render_pass_bundle_indexed_indirect(
        bundle: &mut RenderBundleEncoder,
        buffer_id: id::BufferId,
        offset: BufferAddress,
    ) {
        span!(_guard, TRACE, "RenderBundle::draw_indexed_indirect");
        bundle
            .base
            .commands
            .push(RenderCommand::DrawIndexedIndirect { buffer_id, offset });
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_push_debug_group(
        _bundle: &mut RenderBundleEncoder,
        _label: RawString,
    ) {
        span!(_guard, TRACE, "RenderBundle::push_debug_group");
        //TODO
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_pop_debug_group(_bundle: &mut RenderBundleEncoder) {
        span!(_guard, TRACE, "RenderBundle::pop_debug_group");
        //TODO
    }

    #[no_mangle]
    pub unsafe extern "C" fn wgpu_render_bundle_insert_debug_marker(
        _bundle: &mut RenderBundleEncoder,
        _label: RawString,
    ) {
        span!(_guard, TRACE, "RenderBundle::insert_debug_marker");
        //TODO
    }
}
