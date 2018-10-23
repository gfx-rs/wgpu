use conv;
use registry::{HUB, Items, Registry};
use track::{BufferTracker, TextureTracker};
use {
    Stored,
    CommandBufferId, RenderPassId,
};

use hal;
use hal::command::RawCommandBuffer;


pub struct RenderPass<B: hal::Backend> {
    raw: B::CommandBuffer,
    cmb_id: Stored<CommandBufferId>,
    buffer_tracker: BufferTracker,
    texture_tracker: TextureTracker,
}

impl<B: hal::Backend> RenderPass<B> {
    pub fn new(
        raw: B::CommandBuffer,
        cmb_id: CommandBufferId,
    ) -> Self {
        RenderPass {
            raw,
            cmb_id: Stored(cmb_id),
            buffer_tracker: BufferTracker::new(),
            texture_tracker: TextureTracker::new(),
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_render_pass_end_pass(
    pass_id: RenderPassId,
) -> CommandBufferId {
    let mut pass = HUB.render_passes
        .lock()
        .take(pass_id);
    pass.raw.end_render_pass();

    let buffer_guard = HUB.buffers.lock();
    let texture_guard = HUB.textures.lock();
    let mut cmb_guard = HUB.command_buffers.lock();
    let cmb = cmb_guard.get_mut(pass.cmb_id.0);

    let buffer_barriers = cmb.buffer_tracker
        .consume(pass.buffer_tracker)
        .map(|(id, transit)| {
            let b = buffer_guard.get(id);
            hal::memory::Barrier::Buffer {
                states: conv::map_buffer_state(transit.start) ..
                    conv::map_buffer_state(transit.end),
                target: &b.raw,
            }
        });
    let texture_barriers = cmb.texture_tracker
        .consume(pass.texture_tracker)
        .map(|(id, transit)| {
            let t = texture_guard.get(id);
            hal::memory::Barrier::Image {
                states: conv::map_texture_state(transit.start, t.aspects) ..
                    conv::map_texture_state(transit.end, t.aspects),
                target: &t.raw,
                range: hal::image::SubresourceRange { //TODO!
                    aspects: t.aspects,
                    levels: 0 .. 1,
                    layers: 0 .. 1,
                },
            }
        });

    pass.raw.pipeline_barrier(
        hal::pso::PipelineStage::TOP_OF_PIPE .. hal::pso::PipelineStage::BOTTOM_OF_PIPE,
        hal::memory::Dependencies::empty(),
        buffer_barriers.chain(texture_barriers),
    );

    cmb.raw.push(pass.raw);
    pass.cmb_id.0
}
