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
    device::DeviceError,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{DeviceId, SwapChainId, TextureViewId, Valid},
    resource,
    track::TextureSelector,
    LifeGuard, Stored, SubmissionIndex,
};

use hal::{Device as _, Queue as _, Surface as _};
use std::{borrow::Borrow, marker::PhantomData};
use thiserror::Error;
use wgt::SwapChainStatus as Status;

const FRAME_TIMEOUT_MS: u32 = 1000;
pub const DESIRED_NUM_FRAMES: u32 = 3;

#[derive(Debug)]
pub struct SwapChain<A: hal::Api> {
    pub(crate) life_guard: LifeGuard,
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) desc: wgt::SwapChainDescriptor,
    pub(crate) num_frames: u32,
    pub(crate) acquired_texture: Option<(Stored<TextureViewId>, A::SurfaceTexture)>,
    pub(crate) active_submission_index: SubmissionIndex,
    pub(crate) marker: PhantomData<A>,
}

impl<A: hal::Api> crate::hub::Resource for SwapChain<A> {
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
    #[error("acquired frame is still referenced")]
    StillReferenced,
}

#[derive(Clone, Debug, Error)]
pub enum CreateSwapChainError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("invalid surface")]
    InvalidSurface,
    #[error("`SwapChainOutput` must be dropped before a new `SwapChain` is made")]
    SwapChainOutputExists,
    #[error("Both `SwapChain` width and height must be non-zero. Wait to recreate the `SwapChain` until the window has non-zero area.")]
    ZeroArea,
    #[error("surface does not support the adapter's queue family")]
    UnsupportedQueueFamily,
    #[error("requested format {requested:?} is not in list of supported formats: {available:?}")]
    UnsupportedFormat {
        requested: wgt::TextureFormat,
        available: Vec<wgt::TextureFormat>,
    },
    #[error("requested usage is not supported")]
    UnsupportedUsage,
}

#[repr(C)]
#[derive(Debug)]
pub struct SwapChainOutput {
    pub status: Status,
    pub view_id: Option<TextureViewId>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn swap_chain_get_current_texture_view<A: HalApi>(
        &self,
        swap_chain_id: SwapChainId,
        view_id_in: Input<G, TextureViewId>,
    ) -> Result<SwapChainOutput, SwapChainError> {
        profiling::scope!("get_next_texture", "SwapChain");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.texture_views.prepare(view_id_in);

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(swap_chain_id.to_surface_id())
            .map_err(|_| SwapChainError::InvalidSurface)?;
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
        let sc = swap_chain_guard
            .get_mut(swap_chain_id)
            .map_err(|_| SwapChainError::Invalid)?;

        let device = &device_guard[sc.device_id.value];
        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::GetSwapChainTexture {
                id: fid.id(),
                parent_id: swap_chain_id,
            });
        }

        let suf = A::get_surface_mut(surface);
        let (texture, status) = match unsafe { suf.acquire_texture(FRAME_TIMEOUT_MS) } {
            Ok(Some(ast)) => {
                let status = if ast.suboptimal {
                    Status::Suboptimal
                } else {
                    Status::Good
                };
                (Some(ast.texture), status)
            }
            Ok(None) => (None, Status::Timeout),
            Err(err) => (
                None,
                match err {
                    hal::SurfaceError::Lost => Status::Lost,
                    hal::SurfaceError::Device(err) => {
                        return Err(DeviceError::from(err).into());
                    }
                    hal::SurfaceError::Outdated => Status::Outdated,
                    hal::SurfaceError::Other(msg) => {
                        log::error!("acquire error: {}", msg);
                        Status::Lost
                    }
                },
            ),
        };

        let hal_desc = hal::TextureViewDescriptor {
            label: Some("_Frame"),
            format: sc.desc.format,
            dimension: wgt::TextureViewDimension::D2,
            usage: hal::TextureUse::COLOR_TARGET,
            range: wgt::ImageSubresourceRange::default(),
        };

        let view_id = match texture {
            Some(suf_texture) => {
                let raw = unsafe {
                    device
                        .raw
                        .create_texture_view(suf_texture.borrow(), &hal_desc)
                        .map_err(DeviceError::from)?
                };
                let view = resource::TextureView {
                    raw,
                    source: resource::TextureViewSource::SwapChain(Stored {
                        value: Valid(swap_chain_id),
                        ref_count: sc.life_guard.add_ref(),
                    }),
                    desc: resource::HalTextureViewDescriptor {
                        format: sc.desc.format,
                        dimension: wgt::TextureViewDimension::D2,
                        range: wgt::ImageSubresourceRange::default(),
                    },
                    format_features: wgt::TextureFormatFeatures {
                        allowed_usages: wgt::TextureUsage::RENDER_ATTACHMENT,
                        flags: wgt::TextureFormatFeatureFlags::empty(),
                        filterable: false,
                    },
                    extent: wgt::Extent3d {
                        width: sc.desc.width,
                        height: sc.desc.height,
                        depth_or_array_layers: 1,
                    },
                    samples: 1,
                    sampled_internal_use: hal::TextureUse::empty(),
                    selector: TextureSelector {
                        layers: 0..1,
                        levels: 0..1,
                    },
                    life_guard: LifeGuard::new("<SwapChain View>"),
                };

                let ref_count = view.life_guard.add_ref();
                let id = fid.assign(view, &mut token);

                if sc.acquired_texture.is_some() {
                    return Err(SwapChainError::AlreadyAcquired);
                }

                sc.acquired_texture = Some((
                    Stored {
                        value: id,
                        ref_count,
                    },
                    suf_texture,
                ));

                Some(id.0)
            }
            None => None,
        };

        Ok(SwapChainOutput { status, view_id })
    }

    pub fn swap_chain_present<A: HalApi>(
        &self,
        swap_chain_id: SwapChainId,
    ) -> Result<Status, SwapChainError> {
        profiling::scope!("present", "SwapChain");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(swap_chain_id.to_surface_id())
            .map_err(|_| SwapChainError::InvalidSurface)?;
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut swap_chain_guard, _) = hub.swap_chains.write(&mut token);
        let sc = swap_chain_guard
            .get_mut(swap_chain_id)
            .map_err(|_| SwapChainError::Invalid)?;
        let device = &mut device_guard[sc.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::PresentSwapChain(swap_chain_id));
        }

        let suf_texture = {
            let (view_id, suf_texture) = sc
                .acquired_texture
                .take()
                .ok_or(SwapChainError::AlreadyAcquired)?;

            drop(swap_chain_guard);
            device.suspect_texture_view_for_destruction(view_id.value, &mut token);

            let (mut view_guard, _) = hub.texture_views.write(&mut token);
            let view = &mut view_guard[view_id.value];
            let _ = view.life_guard.ref_count.take();

            suf_texture
        };

        let result = unsafe {
            device
                .queue
                .present(A::get_surface_mut(surface), suf_texture)
        };

        log::debug!("Presented. End of Frame");

        match result {
            Ok(()) => Ok(Status::Good),
            Err(err) => match err {
                hal::SurfaceError::Lost => Ok(Status::Lost),
                hal::SurfaceError::Device(err) => Err(SwapChainError::from(DeviceError::from(err))),
                hal::SurfaceError::Outdated => Ok(Status::Outdated),
                hal::SurfaceError::Other(msg) => {
                    log::error!("acquire error: {}", msg);
                    Err(SwapChainError::InvalidSurface)
                }
            },
        }
    }
}
