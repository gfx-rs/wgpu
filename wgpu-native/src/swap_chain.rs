use crate::{Stored, WeaklyStored,
    DeviceId, SwapChainId, TextureId,
};
use crate::registry::{HUB, Items};
use crate::resource;

use hal;
use hal::{Device as _Device, Swapchain as _Swapchain};

use std::mem;


pub type Epoch = u16;

pub(crate) struct SwapChainLink {
    swap_chain_id: WeaklyStored<SwapChainId>, //TODO: strongly
    epoch: Epoch,
    image_index: hal::SwapImageIndex,
}

pub(crate) struct Surface<B: hal::Backend> {
    pub raw: B::Surface,
}

pub(crate) struct Frame<B: hal::Backend> {
    pub texture: Stored<TextureId>,
    pub fence: B::Fence,
    pub sem_available: B::Semaphore,
    pub sem_present: B::Semaphore,
}

pub(crate) struct SwapChain<B: hal::Backend> {
    pub raw: B::Swapchain,
    pub device_id: Stored<DeviceId>,
    pub frames: Vec<Frame<B>>,
    pub sem_available: B::Semaphore,
    pub epoch: Epoch,
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
    let device_guard = HUB.devices.read();
    let device = device_guard.get(swap_chain.device_id.value);

    let image_index = unsafe {
        let sync = hal::FrameSync::Semaphore(&swap_chain.sem_available);
        swap_chain.raw.acquire_image(!0, sync).unwrap()
    };

    let frame = &mut swap_chain.frames[image_index as usize];
    unsafe {
        device.raw.wait_for_fence(&frame.fence, !0).unwrap();
    }

    mem::swap(&mut frame.sem_available, &mut swap_chain.sem_available);

    let texture_guard = HUB.textures.read();
    let texture = texture_guard.get(frame.texture.value);
    *texture.swap_chain_link.write() = Some(SwapChainLink {
            swap_chain_id: WeaklyStored(swap_chain_id),
            epoch: swap_chain.epoch,
            image_index,
        });

    frame.texture.value
}
