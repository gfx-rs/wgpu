use conv;
use registry::{HUB, Items, Registry};
use track::{BufferTracker, TextureTracker, Tracktion, TrackPermit};
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
    //let texture_guard = HUB.textures.lock();
    let mut cmb_guard = HUB.command_buffers.lock();
    let cmb = cmb_guard.get_mut(pass.cmb_id.0);
    let cmb_buffers = &mut cmb.buffer_tracker;

    let buffer_barriers = pass.buffer_tracker
        .finish()
        .into_iter()
        .flat_map(|(id, new)| {
            match cmb_buffers.track(id.0, new, TrackPermit::REPLACE) {
                Ok(Tracktion::Init) => None,
                Ok(Tracktion::Keep) => None,
                Ok(Tracktion::Replace { old }) => Some(hal::memory::Barrier::Buffer {
                    states: conv::map_buffer_state(old) .. conv::map_buffer_state(new),
                    target: &buffer_guard.get(id.0).raw,
                }),
                Ok(Tracktion::Extend { .. }) |
                Err(_) => panic!("Unable to do the resource transition for a pass"),
            }
        });

    pass.raw.pipeline_barrier(
        hal::pso::PipelineStage::TOP_OF_PIPE .. hal::pso::PipelineStage::BOTTOM_OF_PIPE,
        hal::memory::Dependencies::empty(),
        buffer_barriers,
    );

    cmb.raw.push(pass.raw);
    pass.cmb_id.0
}
