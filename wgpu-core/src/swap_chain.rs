/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! Swap chain management.

    ## Lifecycle

    At the low level, the swap chain is using the new simplified model of gfx-rs.

    A swap chain is a separate object that is backend-dependent but shares the index with
    the parent surface, which is backend-independent. This ensures a 1:1 correspondence
    between them.

    `get_next_image()` requests a new image from the surface. It becomes a part of
    `TextureViewInner::SwapChain` of the resulted view. The view is registered in the HUB
    but not in the device tracker.

    The only operation allowed on the view is to be either a color or a resolve attachment.
    It can only be used in one command buffer, which needs to be submitted before presenting.
    Command buffer tracker knows about the view, but only for the duration of recording.
    The view ID is erased from it at the end, so that it's not merged into the device tracker.

    When a swapchain view is used in `begin_render_pass()`, we assume the start and end image
    layouts purely based on whether or not this view was used in this command buffer before.
    It always starts with `Uninitialized` and ends with `Present`, so that no barriers are
    needed when we need to actually present it.

    In `queue_submit()` we make sure to signal the semaphore whenever we render to a swap
    chain view.

    In `present()` we return the swap chain image back and wait on the semaphore.
!*/

use crate::{
    conv,
    hub::{GfxBackend, Global, IdentityFilter, Token},
    id::{DeviceId, SwapChainId, TextureViewId},
    resource,
    Extent3d,
    Features,
    LifeGuard,
    Stored,
};

use hal::{self, device::Device as _, queue::CommandQueue as _, window::PresentationSurface as _};

use smallvec::SmallVec;


const FRAME_TIMEOUT_MS: u64 = 1000;
pub const DESIRED_NUM_FRAMES: u32 = 3;

#[derive(Debug)]
pub struct SwapChain<B: hal::Backend> {
    pub(crate) life_guard: LifeGuard,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) desc: SwapChainDescriptor,
    pub(crate) num_frames: hal::window::SwapImageIndex,
    pub(crate) semaphore: B::Semaphore,
    pub(crate) acquired_view_id: Option<Stored<TextureViewId>>,
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
    pub(crate) fn to_hal(
        &self,
        num_frames: u32,
        features: Features,
    ) -> hal::window::SwapchainConfig {
        let mut config = hal::window::SwapchainConfig::new(
            self.width,
            self.height,
            conv::map_texture_format(self.format, features),
            num_frames,
        );
        //TODO: check for supported
        config.image_usage = conv::map_texture_usage(self.usage, hal::format::Aspects::COLOR);
        config.composite_alpha_mode = hal::window::CompositeAlphaMode::OPAQUE;
        config.present_mode = match self.present_mode {
            PresentMode::NoVsync => hal::window::PresentMode::IMMEDIATE,
            PresentMode::Vsync => hal::window::PresentMode::FIFO,
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
    pub view_id: TextureViewId,
}

#[derive(Debug)]
pub enum SwapChainGetNextTextureError {
    GpuProcessingTimeout,
}

impl<F: IdentityFilter<TextureViewId>> Global<F> {
    pub fn swap_chain_get_next_texture<B: GfxBackend>(
        &self,
        swap_chain_id: SwapChainId,
        view_id_in: F::Input,
    ) -> Result<SwapChainOutput, SwapChainGetNextTextureError> {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = &mut surface_guard[swap_chain_id.to_surface_id()];
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
        let sc = &mut swap_chain_guard[swap_chain_id];
        let device = &device_guard[sc.device_id.value];

        let (image, _) = {
            let suf = B::get_surface_mut(surface);
            match unsafe { suf.acquire_image(FRAME_TIMEOUT_MS * 1_000_000) } {
                Ok(surface_image) => surface_image,
                Err(hal::window::AcquireError::Timeout) => {
                    return Err(SwapChainGetNextTextureError::GpuProcessingTimeout);
                }
                Err(e) => {
                    log::warn!("acquire_image() failed ({:?}), reconfiguring swapchain", e);
                    let desc = sc.desc.to_hal(sc.num_frames, device.features);
                    unsafe {
                        suf.configure_swapchain(&device.raw, desc).unwrap();
                        suf.acquire_image(FRAME_TIMEOUT_MS * 1_000_000).unwrap()
                    }
                }
            }
        };

        let view = resource::TextureView {
            inner: resource::TextureViewInner::SwapChain {
                image,
                source_id: Stored {
                    value: swap_chain_id,
                    ref_count: sc.life_guard.add_ref(),
                },
                framebuffers: SmallVec::new(),
            },
            format: sc.desc.format,
            extent: hal::image::Extent {
                width: sc.desc.width,
                height: sc.desc.height,
                depth: 1,
            },
            samples: 1,
            range: hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                layers: 0 .. 1,
                levels: 0 .. 1,
            },
            life_guard: LifeGuard::new(),
        };
        let ref_count = view.life_guard.add_ref();
        let view_id = hub
            .texture_views
            .register_identity(view_id_in, view, &mut token);

        assert!(
            sc.acquired_view_id.is_none(),
            "Swap chain image is already acquired"
        );
        sc.acquired_view_id = Some(Stored {
            value: view_id,
            ref_count,
        });

        Ok(SwapChainOutput { view_id })
    }

    pub fn swap_chain_present<B: GfxBackend>(&self, swap_chain_id: SwapChainId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = &mut surface_guard[swap_chain_id.to_surface_id()];
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
        let sc = &mut swap_chain_guard[swap_chain_id];
        let device = &mut device_guard[sc.device_id.value];

        let view_id = sc
            .acquired_view_id
            .take()
            .expect("Swap chain image is not acquired");
        let (view, _) = hub.texture_views.unregister(view_id.value, &mut token);
        let (image, framebuffers) = match view.inner {
            resource::TextureViewInner::Native { .. } => unreachable!(),
            resource::TextureViewInner::SwapChain {
                image,
                framebuffers,
                ..
            } => (image, framebuffers),
        };

        let err = unsafe {
            let queue = &mut device.queue_group.queues[0];
            queue.present_surface(B::get_surface_mut(surface), image, Some(&sc.semaphore))
        };
        if let Err(e) = err {
            log::warn!("present failed: {:?}", e);
        }

        for fbo in framebuffers {
            unsafe {
                device.raw.destroy_framebuffer(fbo);
            }
        }
    }
}
