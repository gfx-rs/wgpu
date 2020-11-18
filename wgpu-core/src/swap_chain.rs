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

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    conv,
    device::DeviceError,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Input, Token},
    id::{DeviceId, SwapChainId, TextureViewId, Valid},
    resource, span,
    track::TextureSelector,
    LifeGuard, PrivateFeatures, Stored, SubmissionIndex,
};

use hal::{self, device::Device as _, queue::CommandQueue as _, window::PresentationSurface as _};
use thiserror::Error;
use wgt::{SwapChainDescriptor, SwapChainStatus};

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
    pub(crate) acquired_framebuffers: Vec<B::Framebuffer>,
    pub(crate) active_submission_index: SubmissionIndex,
}

impl<B: hal::Backend> crate::hub::Resource for SwapChain<B> {
    const TYPE: &'static str = "SwapChain";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
pub enum SwapChainError {
    #[error("swap chain is invalid")]
    Invalid,
    #[error("parent surface is invalid")]
    InvalidSurface,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("swap chain image is already acquired")]
    AlreadyAcquired,
}

#[derive(Clone, Debug, Error)]
pub enum CreateSwapChainError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("invalid surface")]
    InvalidSurface,
    #[error("`SwapChainOutput` must be dropped before a new `SwapChain` is made")]
    SwapChainOutputExists,
    #[error("surface does not support the adapter's queue family")]
    UnsupportedQueueFamily,
    #[error("requested format {requested:?} is not in list of supported formats: {available:?}")]
    UnsupportedFormat {
        requested: hal::format::Format,
        available: Vec<hal::format::Format>,
    },
}

pub(crate) fn swap_chain_descriptor_to_hal(
    desc: &SwapChainDescriptor,
    num_frames: u32,
    private_features: PrivateFeatures,
) -> hal::window::SwapchainConfig {
    let mut config = hal::window::SwapchainConfig::new(
        desc.width,
        desc.height,
        conv::map_texture_format(desc.format, private_features),
        num_frames,
    );
    //TODO: check for supported
    config.image_usage = conv::map_texture_usage(desc.usage, hal::format::Aspects::COLOR);
    config.composite_alpha_mode = hal::window::CompositeAlphaMode::OPAQUE;
    config.present_mode = match desc.present_mode {
        wgt::PresentMode::Immediate => hal::window::PresentMode::IMMEDIATE,
        wgt::PresentMode::Mailbox => hal::window::PresentMode::MAILBOX,
        wgt::PresentMode::Fifo => hal::window::PresentMode::FIFO,
    };
    config
}

#[repr(C)]
#[derive(Debug)]
pub struct SwapChainOutput {
    pub status: SwapChainStatus,
    pub view_id: Option<TextureViewId>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn swap_chain_get_current_texture_view<B: GfxBackend>(
        &self,
        swap_chain_id: SwapChainId,
        view_id_in: Input<G, TextureViewId>,
    ) -> Result<SwapChainOutput, SwapChainError> {
        span!(_guard, INFO, "SwapChain::get_next_texture");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(swap_chain_id.to_surface_id())
            .map_err(|_| SwapChainError::InvalidSurface)?;
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
        let sc = swap_chain_guard
            .get_mut(swap_chain_id)
            .map_err(|_| SwapChainError::Invalid)?;
        #[cfg_attr(not(feature = "trace"), allow(unused_variables))]
        let device = &device_guard[sc.device_id.value];

        let suf = B::get_surface_mut(surface);
        let (image, status) = match unsafe { suf.acquire_image(FRAME_TIMEOUT_MS * 1_000_000) } {
            Ok((surface_image, None)) => (Some(surface_image), SwapChainStatus::Good),
            Ok((surface_image, Some(_))) => (Some(surface_image), SwapChainStatus::Suboptimal),
            Err(err) => (
                None,
                match err {
                    hal::window::AcquireError::OutOfMemory(_) => Err(DeviceError::OutOfMemory)?,
                    hal::window::AcquireError::NotReady => unreachable!(), // we always set a timeout
                    hal::window::AcquireError::Timeout => SwapChainStatus::Timeout,
                    hal::window::AcquireError::OutOfDate => SwapChainStatus::Outdated,
                    hal::window::AcquireError::SurfaceLost(_) => SwapChainStatus::Lost,
                    hal::window::AcquireError::DeviceLost(_) => Err(DeviceError::Lost)?,
                },
            ),
        };

        let view_id = match image {
            Some(image) => {
                let view = resource::TextureView {
                    inner: resource::TextureViewInner::SwapChain {
                        image,
                        source_id: Stored {
                            value: Valid(swap_chain_id),
                            ref_count: sc.life_guard.add_ref(),
                        },
                    },
                    aspects: hal::format::Aspects::COLOR,
                    format: sc.desc.format,
                    extent: hal::image::Extent {
                        width: sc.desc.width,
                        height: sc.desc.height,
                        depth: 1,
                    },
                    samples: 1,
                    selector: TextureSelector {
                        layers: 0..1,
                        levels: 0..1,
                    },
                    life_guard: LifeGuard::new("<SwapChain View>"),
                };

                let ref_count = view.life_guard.add_ref();
                let id = hub
                    .texture_views
                    .register_identity(view_id_in, view, &mut token);

                if sc.acquired_view_id.is_some() {
                    return Err(SwapChainError::AlreadyAcquired);
                }

                sc.acquired_view_id = Some(Stored {
                    value: id,
                    ref_count,
                });

                Some(id.0)
            }
            None => None,
        };

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::GetSwapChainTexture {
                id: view_id,
                parent_id: swap_chain_id,
            });
        }

        Ok(SwapChainOutput { status, view_id })
    }

    pub fn swap_chain_present<B: GfxBackend>(
        &self,
        swap_chain_id: SwapChainId,
    ) -> Result<SwapChainStatus, SwapChainError> {
        span!(_guard, INFO, "SwapChain::present");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(swap_chain_id.to_surface_id())
            .map_err(|_| SwapChainError::InvalidSurface)?;
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
        let sc = swap_chain_guard
            .get_mut(swap_chain_id)
            .map_err(|_| SwapChainError::Invalid)?;
        let device = &mut device_guard[sc.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::PresentSwapChain(swap_chain_id));
        }

        let view_id = sc
            .acquired_view_id
            .take()
            .ok_or(SwapChainError::AlreadyAcquired)?;
        let (view_maybe, _) = hub.texture_views.unregister(view_id.value.0, &mut token);
        let view = view_maybe.ok_or(SwapChainError::Invalid)?;
        let image = match view.inner {
            resource::TextureViewInner::Native { .. } => unreachable!(),
            resource::TextureViewInner::SwapChain { image, .. } => image,
        };

        let sem = if sc.active_submission_index > device.last_completed_submission_index() {
            Some(&sc.semaphore)
        } else {
            None
        };
        let queue = &mut device.queue_group.queues[0];
        let result = unsafe { queue.present(B::get_surface_mut(surface), image, sem) };

        tracing::debug!(trace = true, "Presented. End of Frame");

        for fbo in sc.acquired_framebuffers.drain(..) {
            unsafe {
                device.raw.destroy_framebuffer(fbo);
            }
        }

        match result {
            Ok(None) => Ok(SwapChainStatus::Good),
            Ok(Some(_)) => Ok(SwapChainStatus::Suboptimal),
            Err(err) => match err {
                hal::window::PresentError::OutOfMemory(_) => {
                    Err(SwapChainError::Device(DeviceError::OutOfMemory))
                }
                hal::window::PresentError::OutOfDate => Ok(SwapChainStatus::Outdated),
                hal::window::PresentError::SurfaceLost(_) => Ok(SwapChainStatus::Lost),
                hal::window::PresentError::DeviceLost(_) => {
                    Err(SwapChainError::Device(DeviceError::Lost))
                }
            },
        }
    }
}
