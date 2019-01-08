use hal;

use crate::registry::{HUB, Items};
use crate::resource;
use crate::{Stored,
	SwapChainId, TextureId,
};


pub(crate) struct Surface<B: hal::Backend> {
	pub raw: B::Surface,
}

pub(crate) struct Frame {
	pub texture: Stored<TextureId>,
}

pub(crate) struct SwapChain<B: hal::Backend> {
	pub raw: B::Swapchain,
	pub frames: Vec<Frame>,
	pub next_frame_index: usize,
}

#[repr(C)]
pub struct SwapChainDescriptor {
    pub usage: resource::TextureUsageFlags,
    pub format: resource::TextureFormat,
    pub width: u32,
    pub height: u32,
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_get_next_texture(
    swap_chain_id: SwapChainId,
) -> TextureId {
	let mut swap_chain_guard = HUB.swap_chains.write();
	let swap_chain = swap_chain_guard.get_mut(swap_chain_id);

	let frame = &swap_chain.frames[swap_chain.next_frame_index];
	swap_chain.next_frame_index += 1;
	if swap_chain.next_frame_index == swap_chain.frames.len() {
		swap_chain.next_frame_index = 0;
	}

	//TODO: actual synchronization
	frame.texture.value
}
