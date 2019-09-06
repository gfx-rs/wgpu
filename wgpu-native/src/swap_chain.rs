use crate::{
    conv,
    device::all_image_stages,
    gfx_select,
    hub::{GfxBackend, Token},
    resource,
    DeviceId,
    Extent3d,
    Stored,
    SwapChainId,
    SurfaceId,
    TextureId,
    TextureViewId,
};
#[cfg(not(feature = "remote"))]
use crate::hub::GLOBAL;

use hal::{
    self,
    command::CommandBuffer as _,
    device::Device as _,
    queue::CommandQueue as _,
    window::Swapchain as _,
};
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
    pub image_index: hal::window::SwapImageIndex,
}

impl SwapChainLink<Mutex<SwapImageEpoch>> {
    pub fn bump_epoch(&self) -> SwapImageEpoch {
        let mut epoch = self.epoch.lock();
        *epoch += 1;
        *epoch
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
    pub comb: B::CommandBuffer,
}

//TODO: does it need a ref-counted lifetime?
#[derive(Debug)]
pub struct SwapChain<B: hal::Backend> {
    //Note: it's only an option because we may need to move it out
    // and then put a new swapchain back in.
    pub(crate) raw: Option<B::Swapchain>,
    pub(crate) surface_id: Stored<SurfaceId>,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) desc: SwapChainDescriptor,
    pub(crate) frames: Vec<Frame<B>>,
    pub(crate) acquired: Vec<hal::window::SwapImageIndex>,
    pub(crate) sem_available: B::Semaphore,
    #[cfg_attr(feature = "remote", allow(dead_code))] //TODO: remove
    pub(crate) command_pool: B::CommandPool,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum PresentMode {
    NoVsync = 0,
    Vsync = 1,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct SwapChainDescriptor {
    pub usage: resource::TextureUsage,
    pub format: resource::TextureFormat,
    pub width: u32,
    pub height: u32,
    pub present_mode: PresentMode,
}

impl SwapChainDescriptor {
    pub(crate) fn to_hal(&self, num_frames: u32) -> hal::window::SwapchainConfig {
        let mut config = hal::window::SwapchainConfig::new(
            self.width,
            self.height,
            conv::map_texture_format(self.format),
            num_frames,
        );
        //TODO: check for supported
        config.image_usage = conv::map_texture_usage(self.usage, hal::format::Aspects::COLOR);
        config.composite_alpha = hal::window::CompositeAlpha::OPAQUE;
        config.present_mode = match self.present_mode {
            PresentMode::NoVsync => hal::window::PresentMode::Immediate,
            PresentMode::Vsync => hal::window::PresentMode::Fifo,
        };
        config
    }

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

pub fn swap_chain_get_next_texture<B: GfxBackend>(
    swap_chain_id: SwapChainId
) -> SwapChainOutput {
    let hub = B::hub();
    let mut token = Token::root();

    #[cfg(not(feature = "remote"))]
    let (mut surface_guard, mut token) = GLOBAL.surfaces.write(&mut token);
    let (device_guard, mut token) = hub.devices.read(&mut token);
    let (mut swap_chain_guard, _) = hub.swap_chains.write(&mut token);
    let swap_chain = &mut swap_chain_guard[swap_chain_id];
    let device = &device_guard[swap_chain.device_id.value];

    let image_index = unsafe {
        swap_chain
            .raw
            .as_mut()
            .unwrap()
            .acquire_image(!0, Some(&swap_chain.sem_available), None)
    };

    #[cfg(not(feature = "remote"))]
    {
        if image_index.is_err() {
            warn!("acquire_image failed, re-creating");
            //TODO: remove this once gfx-rs stops destroying the old swapchain
            device.raw.wait_idle().unwrap();
            let (mut texture_guard, mut token) = hub.textures.write(&mut token);
            let (mut texture_view_guard, _) = hub.texture_views.write(&mut token);
            let mut trackers = device.trackers.lock();

            let old_raw = swap_chain.raw.take();
            let config = swap_chain.desc.to_hal(swap_chain.frames.len() as u32);
            let surface = &mut surface_guard[swap_chain.surface_id.value];

            let (raw, images) = unsafe {
                let suf = B::get_surface_mut(surface);
                device
                    .raw
                    .create_swapchain(suf, config, old_raw)
                    .unwrap()
            };
            swap_chain.raw = Some(raw);
            for (frame, image) in swap_chain.frames.iter_mut().zip(images) {
                let texture = &mut texture_guard[frame.texture_id.value];
                let view_raw = unsafe {
                    device
                        .raw
                        .create_image_view(
                            &image,
                            hal::image::ViewKind::D2,
                            conv::map_texture_format(texture.format),
                            hal::format::Swizzle::NO,
                            texture.full_range.clone(),
                        )
                        .unwrap()
                };
                texture.raw = image;
                trackers.textures.reset(
                    frame.texture_id.value,
                    texture.full_range.clone(),
                    resource::TextureUsage::UNINITIALIZED,
                );
                let old_view = mem::replace(&mut texture_view_guard[frame.view_id.value].raw, view_raw);
                unsafe {
                    device.raw.destroy_image_view(old_view);
                }
            }
        }
    }

    let image_index = match image_index {
        Ok((index, suboptimal)) => {
            if suboptimal.is_some() {
                warn!("acquire_image: sub-optimal");
            }
            index
        }
        Err(_) => unsafe {
            swap_chain
                .raw
                .as_mut()
                .unwrap()
                .acquire_image(!0, Some(&swap_chain.sem_available), None)
                .unwrap()
                .0
        },
    };

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

    let (texture_guard, _) = hub.textures.read(&mut token);
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
pub extern "C" fn wgpu_swap_chain_get_next_texture(swap_chain_id: SwapChainId) -> SwapChainOutput {
    gfx_select!(swap_chain_id => swap_chain_get_next_texture(swap_chain_id))
}

pub fn swap_chain_present<B: GfxBackend>(swap_chain_id: SwapChainId) {
    let hub = B::hub();
    let mut token = Token::root();

    let (mut device_guard, mut token) = hub.devices.write(&mut token);
    let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
    let swap_chain = &mut swap_chain_guard[swap_chain_id];
    let device = &mut device_guard[swap_chain.device_id.value];

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

    let (texture_guard, _) = hub.textures.read(&mut token);
    let texture = &texture_guard[frame.texture_id.value];
    texture.placement.as_swap_chain().bump_epoch();

    //TODO: support for swapchain being sampled or read by the shader?

    trace!("transit {:?} to present", frame.texture_id.value);
    let mut trackers = device.trackers.lock();
    let barriers = trackers
        .textures
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
        frame.comb.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        frame.comb.pipeline_barrier(
            all_image_stages() .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            hal::memory::Dependencies::empty(),
            barriers,
        );
        frame.comb.finish();

        // now prepare the GPU submission
        let submission = hal::queue::Submission {
            command_buffers: iter::once(&frame.comb),
            wait_semaphores: None,
            signal_semaphores: Some(&frame.sem_present),
        };

        device.raw.reset_fence(&frame.fence).unwrap();
        let queue = &mut device.queue_group.queues[0];
        queue.submit(submission, Some(&frame.fence));
        queue.present(
            iter::once((swap_chain.raw.as_ref().unwrap(), image_index)),
            iter::once(&frame.sem_present),
        )
    };

    if let Err(e) = err {
        warn!("present failed: {:?}", e);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_swap_chain_present(swap_chain_id: SwapChainId) {
    gfx_select!(swap_chain_id => swap_chain_present(swap_chain_id))
}
