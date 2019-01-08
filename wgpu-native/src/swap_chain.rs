use hal;

use crate::resource;


pub(crate) struct Surface<B: hal::Backend> {
	pub raw: B::Surface,
}

pub(crate) struct SwapChain<B: hal::Backend> {
	pub raw: B::Swapchain,
	pub images: Vec<B::Image>,
}

#[repr(C)]
pub struct SwapChainDescriptor {
    pub usage: resource::TextureUsageFlags,
    pub format: resource::TextureFormat,
    pub width: u32,
    pub height: u32,
}
