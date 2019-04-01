use crate::{
    command::bind::{Binder, LayoutChange},
    conv,
    device::RenderPassContext,
    hub::HUB,
    pipeline::PipelineFlags,
    resource::BufferUsageFlags,
    track::{Stitch, TrackerSet},
    BindGroupId, BufferId, Color, CommandBuffer, CommandBufferId, RenderPassId, RenderPipelineId, Stored,
};

use hal::command::RawCommandBuffer;

use std::{iter, slice};


#[derive(Debug, PartialEq)]
enum BlendColorStatus {
    Unused,
    Required,
    Set,
}

#[derive(Debug, PartialEq)]
enum DrawError {
    MissingBlendColor,
    IncompatibleBindGroup {
        index: u32,
        //expected: BindGroupLayoutId,
        //provided: Option<(BindGroupLayoutId, BindGroupId)>,
    },
}

pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    context: RenderPassContext,
    binder: Binder,
    trackers: TrackerSet,
    blend_color_status: BlendColorStatus,
}

impl<B: hal::Backend> RenderPass<B> {
    pub(crate) fn new(
        raw: B::CommandBuffer,
        cmb_id: Stored<CommandBufferId>,
        context: RenderPassContext,
    ) -> Self {
        RenderPass {
            raw,
            cmb_id,
            context,
            binder: Binder::default(),
            trackers: TrackerSet::new(),
            blend_color_status: BlendColorStatus::Unused,
        }
    }

    fn is_ready(&self) -> Result<(), DrawError> {
        //TODO: vertex buffers
        let bind_mask = self.binder.invalid_mask();
        if bind_mask != 0 {
            //let (expected, provided) = self.binder.entries[index as usize].info();
            return Err(DrawError::IncompatibleBindGroup {
                index: bind_mask.trailing_zeros() as u32,
            });
        }
        if self.blend_color_status == BlendColorStatus::Required {
            return Err(DrawError::MissingBlendColor)
        }
        Ok(())
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(pass_id: RenderPassId) -> CommandBufferId {
    let mut pass = HUB.render_passes.unregister(pass_id);
    unsafe {
        pass.raw.end_render_pass();
    }

    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = &mut cmb_guard[pass.cmb_id.value];

    match cmb.raw.last_mut() {
        Some(ref mut last) => {
            CommandBuffer::insert_barriers(
                last,
                &mut cmb.trackers,
                &pass.trackers,
                Stitch::Last,
                &*HUB.buffers.read(),
                &*HUB.textures.read(),
            );
            unsafe { last.finish() };
        }
        None => {
            cmb.trackers.consume_by_extend(&pass.trackers);
        }
    }

    cmb.raw.push(pass.raw);
    pass.cmb_id.value
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_index_buffer(
    pass_id: RenderPassId,
    buffer_id: BufferId,
    offset: u32,
) {
    let mut pass_guard = HUB.render_passes.write();
    let buffer_guard = HUB.buffers.read();

    let pass = &mut pass_guard[pass_id];
    let buffer = pass
        .trackers
        .buffers
        .get_with_extended_usage(&*buffer_guard, buffer_id, BufferUsageFlags::INDEX)
        .unwrap();

    let view = hal::buffer::IndexBufferView {
        buffer: &buffer.raw,
        offset: offset as u64,
        index_type: hal::IndexType::U16, //TODO?
    };

    unsafe {
        pass.raw.bind_index_buffer(view);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_vertex_buffers(
    pass_id: RenderPassId,
    buffer_ptr: *const BufferId,
    offset_ptr: *const u32,
    count: usize,
) {
    let mut pass_guard = HUB.render_passes.write();
    let buffer_guard = HUB.buffers.read();
    let buffers = unsafe { slice::from_raw_parts(buffer_ptr, count) };
    let offsets = unsafe { slice::from_raw_parts(offset_ptr, count) };

    let pass = &mut pass_guard[pass_id];
    for &id in buffers {
        pass.trackers
            .buffers
            .get_with_extended_usage(&*buffer_guard, id, BufferUsageFlags::VERTEX)
            .unwrap();
    }

    let buffers = buffers
        .iter()
        .map(|&id| &buffer_guard[id].raw)
        .zip(offsets.iter().map(|&off| off as u64));

    unsafe {
        pass.raw.bind_vertex_buffers(0, buffers);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_draw(
    pass_id: RenderPassId,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let mut pass_guard = HUB.render_passes.write();
    let pass = &mut pass_guard[pass_id];
    pass.is_ready().unwrap();

    unsafe {
        pass.raw.draw(
            first_vertex..first_vertex + vertex_count,
            first_instance..first_instance + instance_count,
        );
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_draw_indexed(
    pass_id: RenderPassId,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let mut pass_guard = HUB.render_passes.write();
    let pass = &mut pass_guard[pass_id];
    pass.is_ready().unwrap();

    unsafe {
        pass.raw.draw_indexed(
            first_index..first_index + index_count,
            base_vertex,
            first_instance..first_instance + instance_count,
        );
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_bind_group(
    pass_id: RenderPassId,
    index: u32,
    bind_group_id: BindGroupId,
) {
    let mut pass_guard = HUB.render_passes.write();
    let pass = &mut pass_guard[pass_id];
    let bind_group_guard = HUB.bind_groups.read();
    let bind_group = &bind_group_guard[bind_group_id];

    pass.trackers.consume_by_extend(&bind_group.used);

    if let Some((pipeline_layout_id, follow_up)) =
        pass.binder
            .provide_entry(index as usize, bind_group_id, bind_group)
    {
        let pipeline_layout_guard = HUB.pipeline_layouts.read();
        let bind_groups = iter::once(&bind_group.raw)
            .chain(follow_up.map(|bg_id| &bind_group_guard[bg_id].raw));
        unsafe {
            pass.raw.bind_graphics_descriptor_sets(
                &&pipeline_layout_guard[pipeline_layout_id].raw,
                index as usize,
                bind_groups,
                &[],
            );
        }
    };
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_pipeline(
    pass_id: RenderPassId,
    pipeline_id: RenderPipelineId,
) {
    let mut pass_guard = HUB.render_passes.write();
    let pass = &mut pass_guard[pass_id];
    let pipeline_guard = HUB.render_pipelines.read();
    let pipeline = &pipeline_guard[pipeline_id];

    assert_eq!(
        pass.context, pipeline.pass_context,
        "The render pipeline is not compatible with the pass!"
    );

    if pipeline.flags.contains(PipelineFlags::BLEND_COLOR) && pass.blend_color_status == BlendColorStatus::Unused {
        pass.blend_color_status = BlendColorStatus::Required;
    }

    unsafe {
        pass.raw.bind_graphics_pipeline(&pipeline.raw);
    }

    if pass.binder.pipeline_layout_id == Some(pipeline.layout_id.clone()) {
        return;
    }

    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let pipeline_layout = &pipeline_layout_guard[pipeline.layout_id];
    let bind_group_guard = HUB.bind_groups.read();

    pass.binder.pipeline_layout_id = Some(pipeline.layout_id.clone());
    pass.binder.reset_expectations(pipeline_layout.bind_group_layout_ids.len());

    for (index, (entry, &bgl_id)) in pass
        .binder
        .entries
        .iter_mut()
        .zip(&pipeline_layout.bind_group_layout_ids)
        .enumerate()
    {
        if let LayoutChange::Match(bg_id) = entry.expect_layout(bgl_id) {
            let desc_set = &bind_group_guard[bg_id].raw;
            unsafe {
                pass.raw.bind_graphics_descriptor_sets(
                    &pipeline_layout.raw,
                    index,
                    iter::once(desc_set),
                    &[],
                );
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_blend_color(
    pass_id: RenderPassId,
    color: &Color,
) {
    let mut pass_guard = HUB.render_passes.write();
    let pass = &mut pass_guard[pass_id];

    pass.blend_color_status = BlendColorStatus::Set;

    unsafe {
        pass.raw.set_blend_constants(conv::map_color(color));
    }
}
