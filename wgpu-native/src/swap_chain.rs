use crate::{
    conv,
    device::all_image_stages,
    hub::HUB,
    resource,
    track::TrackPermit,
    DeviceId,
    Extent3d,
    Stored,
    SwapChainId,
    TextureId,
    TextureViewId,
};

use hal::{self, Device as _, Swapchain as _};
use log::{trace, warn};
use parking_lot::Mutex;

use std::{iter, mem};

pub type SwapImageEpoch = u16;

pub(crate) struct SwapChainLink<E> {
    pub swap_chain_id: SwapChainId, //TODO: strongly
    pub epoch: E,
    pub image_index: hal::SwapImageIndex,
}

impl SwapChainLink<Mutex<SwapImageEpoch>> {
    pub fn bump_epoch(&self) -> SwapImageEpoch {
        let mut epoch = self.epoch.lock();
        *epoch += 1;
        *epoch
    }
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
    pub wait_for_epoch: Mutex<Option<SwapImageEpoch>>,
    pub comb: hal::command::CommandBuffer<B, hal::General, hal::command::MultiShot>,
}

//TODO: does it need a ref-counted lifetime?
pub struct SwapChain<B: hal::Backend> {
    pub(crate) raw: B::Swapchain,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) desc: SwapChainDescriptor,
    pub(crate) frames: Vec<Frame<B>>,
    pub(crate) acquired: Vec<hal::SwapImageIndex>,
    pub(crate) sem_available: B::Semaphore,
    #[cfg_attr(not(feature = "local"), allow(dead_code))] //TODO: remove
    pub(crate) command_pool: hal::CommandPool<B, hal::General>,
}

#[repr(C)]
#[derive(Clone, Debug)]
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
pub extern "C" fn wgpu_swap_chain_get_next_texture(swap_chain_id: SwapChainId) -> SwapChainOutput {
    let (image_index, device_id, descriptor) = {
        let mut surface_guard = HUB.surfaces.write();
        let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();
        let sync = hal::FrameSync::Semaphore(&swap_chain.sem_available);
        let result = unsafe { swap_chain.raw.acquire_image(!0, sync) };
        (
            result.ok(),
            swap_chain.device_id.value,
            swap_chain.desc.clone(),
        )
    };

    #[cfg(not(feature = "local"))]
    let _ = descriptor;
    #[cfg(feature = "local")]
    {
        use crate::device::{device_create_swap_chain, swap_chain_populate_textures};
        if image_index.is_none() {
            warn!("acquire_image failed, re-creating");
            let textures = device_create_swap_chain(device_id, swap_chain_id, &descriptor);
            swap_chain_populate_textures(swap_chain_id, textures);
        }
    }

    let mut surface_guard = HUB.surfaces.write();
    let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();

    let image_index = match image_index {
        Some(index) => index,
        None => {
            let sync = hal::FrameSync::Semaphore(&swap_chain.sem_available);
            unsafe { swap_chain.raw.acquire_image(!0, sync) }.unwrap()
        }
    };

    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];

    assert_ne!(
        swap_chain.acquired.len(),
        swap_chain.acquired.capacity(),
        "Unable to acquire any more swap chain images before presenting"
    );
    swap_chain.acquired.push(image_index);

    let frame = &mut swap_chain.frames[image_index as usize];
    unsafe {
        device.raw.wait_for_fence(&frame.fence, !0).unwrap();
    }
    mem::swap(&mut frame.sem_available, &mut swap_chain.sem_available);

    let frame_epoch = HUB.textures.read()[frame.texture_id.value]
        .placement
        .as_swap_chain()
        .bump_epoch();

    *frame.wait_for_epoch.lock() = Some(frame_epoch);

    SwapChainOutput {
        texture_id: frame.texture_id.value,
        view_id: frame.view_id.value,
    }
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_present(swap_chain_id: SwapChainId) {
    let mut surface_guard = HUB.surfaces.write();
    let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();

    let image_index = swap_chain.acquired.remove(0);
    let frame = &mut swap_chain.frames[image_index as usize];

    let mut device_guard = HUB.devices.write();
    let device = &mut device_guard[swap_chain.device_id.value];

    let texture_guard = HUB.textures.read();
    let texture = &texture_guard[frame.texture_id.value];
    texture.placement.as_swap_chain().bump_epoch();

    //TODO: support for swapchain being sampled or read by the shader?

    trace!("transit {:?} to present", frame.texture_id.value);
    let barrier = device
        .trackers
        .lock()
        .textures
        .transit(
            frame.texture_id.value,
            &texture.life_guard.ref_count,
            resource::TextureUsageFlags::UNINITIALIZED,
            TrackPermit::REPLACE,
        )
        .unwrap()
        .into_source()
        .map(|old| hal::memory::Barrier::Image {
            states: conv::map_texture_state(old, hal::format::Aspects::COLOR)
                ..(
                    hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::Present,
                ),
            target: &texture.raw,
            families: None,
            range: texture.full_range.clone(),
        });

    let err = unsafe {
        frame.comb.begin(false);
        frame.comb.pipeline_barrier(
            all_image_stages()..hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
