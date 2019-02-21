use crate::{
    Extent3d, Stored, WeaklyStored,
    DeviceId, SwapChainId, TextureId, TextureViewId,
};
use crate::{conv, resource};
use crate::device::all_image_stages;
use crate::hub::HUB;
use crate::track::{TrackPermit};

use hal;
use hal::{Device as _Device, Swapchain as _Swapchain};
use log::{trace, warn};

use std::{iter, mem};


pub type SwapImageEpoch = u16;

pub(crate) struct SwapChainLink<E> {
    pub swap_chain_id: WeaklyStored<SwapChainId>, //TODO: strongly
    pub epoch: E,
    pub image_index: hal::SwapImageIndex,
}

pub struct Surface<B: hal::Backend> {
    pub(crate) raw: B::Surface,
    pub(crate) swap_chain: Option<SwapChain<B>>,
}

impl<B: hal::Backend> Surface<B> {
    pub(crate) fn new(raw: B::Surface) -> Self {
        Surface {
            raw,
            swap_chain: None,
        }
    }
}

pub(crate) struct Frame<B: hal::Backend> {
    pub texture_id: Stored<TextureId>,
    pub view_id: Stored<TextureViewId>,
    pub fence: B::Fence,
    pub sem_available: B::Semaphore,
    pub sem_present: B::Semaphore,
    pub comb: hal::command::CommandBuffer<B, hal::General, hal::command::MultiShot>,
}

pub struct OutdatedFrame {
    pub(crate) texture_id: Stored<TextureId>,
    pub(crate) view_id: Stored<TextureViewId>,
}

const OUTDATED_IMAGE_INDEX: u32 = !0;
//TODO: does it need a ref-counted lifetime?

pub struct SwapChain<B: hal::Backend> {
    pub(crate) raw: B::Swapchain,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) frames: Vec<Frame<B>>,
    pub(crate) acquired: Vec<hal::SwapImageIndex>,
    pub(crate) sem_available: B::Semaphore,
    pub(crate) outdated: OutdatedFrame,
    #[cfg_attr(not(feature = "local"), allow(dead_code))] //TODO: remove
    pub(crate) command_pool: hal::CommandPool<B, hal::General>,
}

#[repr(C)]
pub struct SwapChainDescriptor {
    pub usage: resource::TextureUsageFlags,
    pub format: resource::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl SwapChainDescriptor {
    pub fn to_texture_desc(&self) -> resource::TextureDescriptor {
        resource::TextureDescriptor {
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth: 1,
            },
            array_size: 1,
            dimension: resource::TextureDimension::D2,
            format: self.format,
            usage: self.usage,
        }
    }
}

#[repr(C)]
pub struct SwapChainOutput {
    pub texture_id: TextureId,
    pub view_id: TextureViewId,
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_get_next_texture(
    swap_chain_id: SwapChainId,
) -> SwapChainOutput {
    let mut surface_guard = HUB.surfaces.write();
    let swap_chain = surface_guard
        .get_mut(swap_chain_id)
        .swap_chain
        .as_mut()
        .unwrap();
    let device_guard = HUB.devices.read();
    let device = device_guard.get(swap_chain.device_id.value);

    assert_ne!(swap_chain.acquired.len(), swap_chain.acquired.capacity(),
        "Unable to acquire any more swap chain images before presenting");

    match {
        let sync = hal::FrameSync::Semaphore(&swap_chain.sem_available);
        unsafe { swap_chain.raw.acquire_image(!0, sync) }
    } {
        Ok(image_index) => {
            swap_chain.acquired.push(image_index);
            let frame = &mut swap_chain.frames[image_index as usize];
            unsafe {
                device.raw.wait_for_fence(&frame.fence, !0).unwrap();
            }

            mem::swap(&mut frame.sem_available, &mut swap_chain.sem_available);

            let texture_guard = HUB.textures.read();
            let texture = texture_guard.get(frame.texture_id.value);
            match texture.swap_chain_link {
                Some(ref link) => *link.epoch.lock() += 1,
                None => unreachable!(),
            }

            SwapChainOutput {
                texture_id: frame.texture_id.value,
                view_id: frame.view_id.value,
            }
        }
        Err(e) => {
            warn!("acquire_image failed: {:?}", e);
            swap_chain.acquired.push(OUTDATED_IMAGE_INDEX);
            SwapChainOutput {
                texture_id: swap_chain.outdated.texture_id.value,
                view_id: swap_chain.outdated.view_id.value,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_present(
    swap_chain_id: SwapChainId,
) {
    let mut surface_guard = HUB.surfaces.write();
    let swap_chain = surface_guard
        .get_mut(swap_chain_id)
        .swap_chain
        .as_mut()
        .unwrap();

    let image_index = swap_chain.acquired.remove(0);
    let frame = match swap_chain.frames.get_mut(image_index as usize) {
        Some(frame) => frame,
        None => {
            assert_eq!(image_index, OUTDATED_IMAGE_INDEX);
            return
        }
    };

    let mut device_guard = HUB.devices.write();
    let device = device_guard.get_mut(swap_chain.device_id.value);

    let texture_guard = HUB.textures.read();
    let texture = texture_guard.get(frame.texture_id.value);
    match texture.swap_chain_link {
        Some(ref link) => *link.epoch.lock() += 1,
        None => unreachable!(),
    }

    //TODO: support for swapchain being sampled or read by the shader?

    trace!("transit {:?} to present", frame.texture_id.value);
    let barrier = device.texture_tracker
        .lock()
        .transit(
            frame.texture_id.value,
            &texture.life_guard.ref_count,
            resource::TextureUsageFlags::UNINITIALIZED,
            TrackPermit::REPLACE,
        )
        .unwrap()
        .into_source()
        .map(|old| hal::memory::Barrier::Image {
            states: conv::map_texture_state(old, hal::format::Aspects::COLOR) ..
                (hal::image::Access::COLOR_ATTACHMENT_WRITE, hal::image::Layout::Present),
            target: &texture.raw,
            families: None,
            range: texture.full_range.clone(),
        });

    let err = unsafe {
        frame.comb.begin(false);
        frame.comb.pipeline_barrier(
            all_image_stages() .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            hal::memory::Dependencies::empty(),
            barrier,
        );
        frame.comb.finish();

        // now prepare the GPU submission
        let submission = hal::Submission {
            command_buffers: iter::once(&frame.comb),
            wait_semaphores: None,
            signal_semaphores: Some(&frame.sem_present),
        };

        device.raw.reset_fence(&frame.fence).unwrap();
        let queue = &mut device.queue_group.queues[0];
        queue.submit(submission, Some(&frame.fence));
        queue.present(
            iter::once((&swap_chain.raw, image_index)),
            iter::once(&frame.sem_present),
        )
    };

    if let Err(e) = err {
        warn!("present failed: {:?}", e);
    }
}
