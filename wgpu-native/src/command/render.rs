use crate::command::bind::Binder;
use crate::resource::BufferUsageFlags;
use crate::registry::{Items, HUB};
use crate::track::{BufferTracker, TextureTracker, TrackPermit};
use crate::{
    CommandBuffer, Stored,
    BindGroupId, BufferId, CommandBufferId, RenderPassId, RenderPipelineId,
};

use hal::command::RawCommandBuffer;

use std::iter;


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    binder: Binder,
    buffer_tracker: BufferTracker,
    texture_tracker: TextureTracker,
}

impl<B: hal::Backend> RenderPass<B> {
    pub(crate) fn new(raw: B::CommandBuffer, cmb_id: Stored<CommandBufferId>) -> Self {
        RenderPass {
            raw,
            cmb_id,
            binder: Binder::default(),
            buffer_tracker: BufferTracker::new(),
            texture_tracker: TextureTracker::new(),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(pass_id: RenderPassId) -> CommandBufferId {
    let mut pass = HUB.render_passes.write().take(pass_id);
    unsafe {
        pass.raw.end_render_pass();
    }

    let mut cmb_guard = HUB.command_buffers.write();
    let cmb = cmb_guard.get_mut(pass.cmb_id.value);

    if let Some(ref mut last) = cmb.raw.last_mut() {
        CommandBuffer::insert_barriers(
            last,
            cmb.buffer_tracker.consume_by_replace(&pass.buffer_tracker),
            cmb.texture_tracker.consume_by_replace(&pass.texture_tracker),
            &*HUB.buffers.read(),
            &*HUB.textures.read(),
        );
        unsafe { last.finish() };
    }

    cmb.raw.push(pass.raw);
    pass.cmb_id.value
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_index_buffer(
    pass_id: RenderPassId, buffer_id: BufferId, offset: u32
) {
    let mut pass_guard = HUB.render_passes.write();
    let buffer_guard = HUB.buffers.read();

    let pass = pass_guard.get_mut(pass_id);
    let (buffer, _) = pass.buffer_tracker
        .get_with_usage(
            &*buffer_guard,
            buffer_id,
            BufferUsageFlags::INDEX,
            TrackPermit::EXTEND,
        )
        .unwrap();
        buffer_guard.get(buffer_id);

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
    pass_id: RenderPassId, buffers: &[BufferId], offsets: &[u32]
) {
    let mut pass_guard = HUB.render_passes.write();
    let buffer_guard = HUB.buffers.read();

    let pass = pass_guard.get_mut(pass_id);
    for &id in buffers {
        pass.buffer_tracker
            .get_with_usage(
                &*buffer_guard,
                id,
                BufferUsageFlags::VERTEX,
                TrackPermit::EXTEND,
            )
            .unwrap();
    }

    assert_eq!(buffers.len(), offsets.len());
    let buffers = buffers
        .iter()
        .map(|&id| &buffer_guard.get(id).raw)
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
    unsafe {
        HUB.render_passes
            .write()
            .get_mut(pass_id)
            .raw
            .draw(
                first_vertex .. first_vertex + vertex_count,
                first_instance .. first_instance + instance_count,
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
    unsafe {
        HUB.render_passes
            .write()
            .get_mut(pass_id)
            .raw
            .draw_indexed(
                first_index .. first_index + index_count,
                base_vertex,
                first_instance .. first_instance + instance_count,
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
    let pass = pass_guard.get_mut(pass_id);
    let bind_group_guard = HUB.bind_groups.read();
    let bind_group = bind_group_guard.get(bind_group_id);

    pass.buffer_tracker
        .consume_by_extend(&bind_group.used_buffers)
        .unwrap();
    pass.texture_tracker
        .consume_by_extend(&bind_group.used_textures)
        .unwrap();

    if let Some(pipeline_layout_id) = pass.binder.provide_entry(index as usize, bind_group_id, bind_group) {
        let pipeline_layout_guard = HUB.pipeline_layouts.read();
        let pipeline_layout = pipeline_layout_guard.get(pipeline_layout_id);
        unsafe {
            pass.raw.bind_graphics_descriptor_sets(
                &pipeline_layout.raw,
                index as usize,
                iter::once(&bind_group.raw),
                &[],
            );
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_set_pipeline(
    pass_id: RenderPassId,
    pipeline_id: RenderPipelineId,
) {
    let mut pass_guard = HUB.render_passes.write();
    let pass = pass_guard.get_mut(pass_id);
    let pipeline_guard = HUB.render_pipelines.read();
    let pipeline = pipeline_guard.get(pipeline_id);

    unsafe {
        pass.raw.bind_graphics_pipeline(&pipeline.raw);
    }

    if pass.binder.pipeline_layout_id == Some(pipeline.layout_id.clone()) {
        return
    }

    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let pipeline_layout = pipeline_layout_guard.get(pipeline.layout_id.0);
    let bing_group_guard = HUB.bind_groups.read();

    pass.binder.pipeline_layout_id = Some(pipeline.layout_id.clone());
    pass.binder.ensure_length(pipeline_layout.bind_group_layout_ids.len());

    for (index, (entry, bgl_id)) in pass.binder.entries
        .iter_mut()
        .zip(&pipeline_layout.bind_group_layout_ids)
        .enumerate()
    {
        if let Some(bg_id) = entry.expect_layout(bgl_id.0) {
            let bind_group = bing_group_guard.get(bg_id);
            unsafe {
                pass.raw.bind_graphics_descriptor_sets(
                    &pipeline_layout.raw,
                    index,
                    iter::once(&bind_group.raw),
                    &[]
                );
            }
        }
    }
}
