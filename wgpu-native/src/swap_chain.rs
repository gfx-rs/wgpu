use crate::{
    conv,
    device::all_image_stages,
    hub::{HUB, Token},
    resource,
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

use std::{
    iter,
    mem,
    sync::atomic::{AtomicBool, Ordering},
};

pub type SwapImageEpoch = u64;

const FRAME_TIMEOUT_MS: u64 = 1000;

#[derive(Debug)]
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

#[derive(Debug)]
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

#[derive(Debug)]
pub(crate) struct Frame<B: hal::Backend> {
    pub texture_id: Stored<TextureId>,
    pub view_id: Stored<TextureViewId>,
    pub fence: B::Fence,
    pub sem_available: B::Semaphore,
    pub sem_present: B::Semaphore,
    pub acquired_epoch: Option<SwapImageEpoch>,
    pub need_waiting: AtomicBool,
    pub comb: hal::command::CommandBuffer<B, hal::General, hal::command::MultiShot>,
}

//TODO: does it need a ref-counted lifetime?
#[derive(Debug)]
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
    pub usage: resource::TextureUsage,
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
            mip_level_count: 1,
            array_layer_count: 1,
            sample_count: 1,
            dimension: resource::TextureDimension::D2,
            format: self.format,
            usage: self.usage,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SwapChainOutput {
    pub texture_id: TextureId,
    pub view_id: TextureViewId,
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_get_next_texture(swap_chain_id: SwapChainId) -> SwapChainOutput {
    let mut token = Token::root();
    let (image_index, device_id, descriptor) = {
        let (mut surface_guard, _) = HUB.surfaces.write(&mut token);
        let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();
        let result = unsafe {
            swap_chain
                .raw
                .acquire_image(!0, Some(&swap_chain.sem_available), None)
        };
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
            let textures = device_create_swap_chain(device_id, swap_chain_id, &descriptor, &mut token);
            swap_chain_populate_textures(swap_chain_id, textures, &mut token);
        }
    }

    let (mut surface_guard, mut token) = HUB.surfaces.write(&mut token);
    let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();

    let image_index = match image_index {
        Some((index, suboptimal)) => {
            if suboptimal.is_some() {
                warn!("acquire_image: sub-optimal");
            }
            index
        }
        None => unsafe {
            swap_chain
                .raw
                .acquire_image(!0, Some(&swap_chain.sem_available), None)
                .unwrap()
                .0
        },
    };

    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let device = &device_guard[device_id];

    assert_ne!(
        swap_chain.acquired.len(),
        swap_chain.acquired.capacity(),
        "Unable to acquire any more swap chain images before presenting"
    );
    swap_chain.acquired.push(image_index);

    let frame = &mut swap_chain.frames[image_index as usize];
    let status = unsafe {
        device
            .raw
            .wait_for_fence(&frame.fence, FRAME_TIMEOUT_MS * 1_000_000)
    };
    assert_eq!(
        status,
        Ok(true),
        "GPU got stuck on a frame (image {}) :(",
        image_index
    );
    mem::swap(&mut frame.sem_available, &mut swap_chain.sem_available);
    frame.need_waiting.store(true, Ordering::Release);

    let (texture_guard, _) = HUB.textures.read(&mut token);
    let frame_epoch = texture_guard[frame.texture_id.value]
        .placement
        .as_swap_chain()
        .bump_epoch();

    assert_eq!(
        frame.acquired_epoch, None,
        "Last swapchain output hasn't been presented"
    );
    frame.acquired_epoch = Some(frame_epoch);

    SwapChainOutput {
        texture_id: frame.texture_id.value,
        view_id: frame.view_id.value,
    }
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_present(swap_chain_id: SwapChainId) {
    let mut token = Token::root();
    let (mut surface_guard, mut token) = HUB.surfaces.write(&mut token);
    let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();

    let image_index = swap_chain.acquired.remove(0);
    let frame = &mut swap_chain.frames[image_index as usize];
    let epoch = frame.acquired_epoch.take();
    assert!(
        epoch.is_some(),
        "Presented frame (image {}) was not acquired",
        image_index
    );
    assert!(
        !frame.need_waiting.load(Ordering::Acquire),
        "No rendering work has been submitted for the presented frame (image {})",
        image_index
    );

    let (mut device_guard, mut token) = HUB.devices.write(&mut token);
    let device = &mut device_guard[swap_chain.device_id.value];

    let (texture_guard, _) = HUB.textures.read(&mut token);
    let texture = &texture_guard[frame.texture_id.value];
    texture.placement.as_swap_chain().bump_epoch();

    //TODO: support for swapchain being sampled or read by the shader?

    trace!("transit {:?} to present", frame.texture_id.value);
    let mut trackers = device.trackers.lock();
    let barriers = trackers.textures
        .change_replace(
            frame.texture_id.value,
            &texture.life_guard.ref_count,
            texture.full_range.clone(),
            resource::TextureUsage::UNINITIALIZED,
        )
        .map(|pending| hal::memory::Barrier::Image {
            states: conv::map_texture_state(pending.usage.start, hal::format::Aspects::COLOR)
                .. (
                    hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::Present,
                ),
            target: &texture.raw,
            families: None,
            range: pending.selector,
        });

    let err = unsafe {
        frame.comb.begin(false);
        frame.comb.pipeline_barrier(
            all_image_stages() .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            hal::memory::Dependencies::empty(),
            barriers,
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
