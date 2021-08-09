/*! Presentation.

    ## Lifecycle

    When a swapchain view is used in `begin_render_pass()`, we assume the start and end image
    layouts purely based on whether or not this view was used in this command buffer before.
    It always starts with `Uninitialized` and ends with `Present`, so that no barriers are
    needed when we need to actually present it.
!*/

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    device::DeviceError,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{DeviceId, SurfaceId, TextureViewId, Valid},
    resource,
    track::TextureSelector,
    LifeGuard, Stored, SubmissionIndex,
};

use hal::{Device as _, Queue as _, Surface as _};
use std::borrow::Borrow;
use thiserror::Error;
use wgt::SurfaceStatus as Status;

const FRAME_TIMEOUT_MS: u32 = 1000;
pub const DESIRED_NUM_FRAMES: u32 = 3;

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) config: wgt::SurfaceConfiguration,
    pub(crate) num_frames: u32,
    pub(crate) acquired_texture: Option<Stored<TextureViewId>>,
    pub(crate) active_submission_index: SubmissionIndex,
}

#[derive(Clone, Debug, Error)]
pub enum SurfaceError {
    #[error("surface is invalid")]
    Invalid,
    #[error("surface is not configured for presentation")]
    NotConfigured,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("surface image is already acquired")]
    AlreadyAcquired,
    #[error("acquired frame is still referenced")]
    StillReferenced,
}

#[derive(Clone, Debug, Error)]
pub enum ConfigureSurfaceError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("invalid surface")]
    InvalidSurface,
    #[error("`SurfaceOutput` must be dropped before a new `Surface` is made")]
    PreviousOutputExists,
    #[error("Both `Surface` width and height must be non-zero. Wait to recreate the `Surface` until the window has non-zero area.")]
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
pub struct SurfaceOutput {
    pub status: Status,
    pub view_id: Option<TextureViewId>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn surface_get_current_texture_view<A: HalApi>(
        &self,
        surface_id: SurfaceId,
        view_id_in: Input<G, TextureViewId>,
    ) -> Result<SurfaceOutput, SurfaceError> {
        profiling::scope!("get_next_texture", "SwapChain");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.texture_views.prepare(view_id_in);

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(surface_id)
            .map_err(|_| SurfaceError::Invalid)?;
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let device = match surface.presentation {
            Some(ref present) => &device_guard[present.device_id.value],
            None => return Err(SurfaceError::NotConfigured),
        };

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::GetSurfaceTexture {
                id: fid.id(),
                parent_id: surface_id,
            });
        }

        let config = surface.presentation.as_ref().unwrap().config.clone();
        let suf = A::get_surface_mut(surface);
        let (texture, status) = match unsafe { suf.raw.acquire_texture(FRAME_TIMEOUT_MS) } {
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
            format: config.format,
            dimension: wgt::TextureViewDimension::D2,
            usage: hal::TextureUses::COLOR_TARGET,
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
                    source: resource::TextureViewSource::Surface(Valid(surface_id)),
                    desc: resource::HalTextureViewDescriptor {
                        format: config.format,
                        dimension: wgt::TextureViewDimension::D2,
                        range: wgt::ImageSubresourceRange::default(),
                    },
                    format_features: wgt::TextureFormatFeatures {
                        allowed_usages: wgt::TextureUsages::RENDER_ATTACHMENT,
                        flags: wgt::TextureFormatFeatureFlags::empty(),
                        filterable: false,
                    },
                    extent: wgt::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    samples: 1,
                    sampled_internal_use: hal::TextureUses::empty(),
                    selector: TextureSelector {
                        layers: 0..1,
                        levels: 0..1,
                    },
                    life_guard: LifeGuard::new("<SwapChain View>"),
                };

                let ref_count = view.life_guard.add_ref();
                let id = fid.assign(view, &mut token);

                suf.acquired_texture = Some(suf_texture);

                let present = surface.presentation.as_mut().unwrap();
                if present.acquired_texture.is_some() {
                    return Err(SurfaceError::AlreadyAcquired);
                }
                present.acquired_texture = Some(Stored {
                    value: id,
                    ref_count,
                });

                Some(id.0)
            }
            None => None,
        };

        Ok(SurfaceOutput { status, view_id })
    }

    pub fn surface_present<A: HalApi>(
        &self,
        surface_id: SurfaceId,
    ) -> Result<Status, SurfaceError> {
        profiling::scope!("present", "SwapChain");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(surface_id)
            .map_err(|_| SurfaceError::Invalid)?;
        let (mut device_guard, mut token) = hub.devices.write(&mut token);

        let present = match surface.presentation {
            Some(ref mut present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = &mut device_guard[present.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::Present(surface_id));
        }

        let result = {
            let view_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;
            let suf = A::get_surface_mut(surface);
            let suf_texture = suf.acquired_texture.take().unwrap();

            let result = unsafe { device.queue.present(&mut suf.raw, suf_texture) };

            drop(surface_guard);

            let (view, _) = hub.texture_views.unregister(view_id.value.0, &mut token);
            if let Some(view) = view {
                device.schedule_rogue_texture_view_for_destruction(view_id.value, view, &mut token);
            }

            result
        };

        log::debug!("Presented. End of Frame");

        match result {
            Ok(()) => Ok(Status::Good),
            Err(err) => match err {
                hal::SurfaceError::Lost => Ok(Status::Lost),
                hal::SurfaceError::Device(err) => Err(SurfaceError::from(DeviceError::from(err))),
                hal::SurfaceError::Outdated => Ok(Status::Outdated),
                hal::SurfaceError::Other(msg) => {
                    log::error!("acquire error: {}", msg);
                    Err(SurfaceError::Invalid)
                }
            },
        }
    }
}
