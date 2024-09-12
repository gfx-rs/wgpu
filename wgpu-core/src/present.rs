/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use std::{mem::ManuallyDrop, sync::Arc};

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    conv,
    device::{Device, DeviceError, MissingDownlevelFlags, WaitIdleError},
    global::Global,
    hal_label, id,
    resource::{self, Trackable},
};

use thiserror::Error;
use wgt::SurfaceStatus as Status;

const FRAME_TIMEOUT_MS: u32 = 1000;

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device: Arc<Device>,
    pub(crate) config: wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
    pub(crate) acquired_texture: Option<id::TextureId>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum SurfaceError {
    #[error("Surface is invalid")]
    Invalid,
    #[error("Surface is not configured for presentation")]
    NotConfigured,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Surface image is already acquired")]
    AlreadyAcquired,
    #[error("Acquired frame is still referenced")]
    StillReferenced,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ConfigureSurfaceError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Invalid surface")]
    InvalidSurface,
    #[error("The view format {0:?} is not compatible with texture format {1:?}, only changing srgb-ness is allowed.")]
    InvalidViewFormat(wgt::TextureFormat, wgt::TextureFormat),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
    #[error("`SurfaceOutput` must be dropped before a new `Surface` is made")]
    PreviousOutputExists,
    #[error("Both `Surface` width and height must be non-zero. Wait to recreate the `Surface` until the window has non-zero area.")]
    ZeroArea,
    #[error("`Surface` width and height must be within the maximum supported texture size. Requested was ({width}, {height}), maximum extent for either dimension is {max_texture_dimension_2d}.")]
    TooLarge {
        width: u32,
        height: u32,
        max_texture_dimension_2d: u32,
    },
    #[error("Surface does not support the adapter's queue family")]
    UnsupportedQueueFamily,
    #[error("Requested format {requested:?} is not in list of supported formats: {available:?}")]
    UnsupportedFormat {
        requested: wgt::TextureFormat,
        available: Vec<wgt::TextureFormat>,
    },
    #[error("Requested present mode {requested:?} is not in the list of supported present modes: {available:?}")]
    UnsupportedPresentMode {
        requested: wgt::PresentMode,
        available: Vec<wgt::PresentMode>,
    },
    #[error("Requested alpha mode {requested:?} is not in the list of supported alpha modes: {available:?}")]
    UnsupportedAlphaMode {
        requested: wgt::CompositeAlphaMode,
        available: Vec<wgt::CompositeAlphaMode>,
    },
    #[error("Requested usage {requested:?} is not in the list of supported usages: {available:?}")]
    UnsupportedUsage {
        requested: hal::TextureUses,
        available: hal::TextureUses,
    },
    #[error("Gpu got stuck :(")]
    StuckGpu,
}

impl From<WaitIdleError> for ConfigureSurfaceError {
    fn from(e: WaitIdleError) -> Self {
        match e {
            WaitIdleError::Device(d) => ConfigureSurfaceError::Device(d),
            WaitIdleError::WrongSubmissionIndex(..) => unreachable!(),
            WaitIdleError::StuckGpu => ConfigureSurfaceError::StuckGpu,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SurfaceOutput {
    pub status: Status,
    pub texture_id: Option<id::TextureId>,
}

impl Global {
    pub fn surface_get_current_texture(
        &self,
        surface_id: id::SurfaceId,
        texture_id_in: Option<id::TextureId>,
    ) -> Result<SurfaceOutput, SurfaceError> {
        profiling::scope!("SwapChain::get_next_texture");

        let hub = &self.hub;

        let surface = self.surfaces.get(surface_id);

        let (device, config) = if let Some(ref present) = *surface.presentation.lock() {
            present.device.check_is_valid()?;
            (present.device.clone(), present.config.clone())
        } else {
            return Err(SurfaceError::NotConfigured);
        };

        let fid = hub.textures.prepare(texture_id_in);

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::GetSurfaceTexture {
                id: fid.id(),
                parent_id: surface_id,
            });
        }

        let fence = device.fence.read();

        let suf = surface.raw(device.backend()).unwrap();
        let (texture_id, status) = match unsafe {
            suf.acquire_texture(
                Some(std::time::Duration::from_millis(FRAME_TIMEOUT_MS as u64)),
                fence.as_ref(),
            )
        } {
            Ok(Some(ast)) => {
                drop(fence);

                let texture_desc = wgt::TextureDescriptor {
                    label: Some(std::borrow::Cow::Borrowed("<Surface Texture>")),
                    size: wgt::Extent3d {
                        width: config.width,
                        height: config.height,
                        depth_or_array_layers: 1,
                    },
                    sample_count: 1,
                    mip_level_count: 1,
                    format: config.format,
                    dimension: wgt::TextureDimension::D2,
                    usage: config.usage,
                    view_formats: config.view_formats,
                };
                let hal_usage = conv::map_texture_usage(config.usage, config.format.into());
                let format_features = wgt::TextureFormatFeatures {
                    allowed_usages: wgt::TextureUsages::RENDER_ATTACHMENT,
                    flags: wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4
                        | wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
                };
                let clear_view_desc = hal::TextureViewDescriptor {
                    label: hal_label(
                        Some("(wgpu internal) clear surface texture view"),
                        device.instance_flags,
                    ),
                    format: config.format,
                    dimension: wgt::TextureViewDimension::D2,
                    usage: hal::TextureUses::COLOR_TARGET,
                    range: wgt::ImageSubresourceRange::default(),
                };
                let clear_view = unsafe {
                    device
                        .raw()
                        .create_texture_view(ast.texture.as_ref().borrow(), &clear_view_desc)
                }
                .map_err(|e| device.handle_hal_error(e))?;

                let mut presentation = surface.presentation.lock();
                let present = presentation.as_mut().unwrap();
                let texture = resource::Texture::new(
                    &device,
                    resource::TextureInner::Surface {
                        raw: ast.texture,
                        parent_id: surface_id,
                    },
                    hal_usage,
                    &texture_desc,
                    format_features,
                    resource::TextureClearMode::Surface {
                        clear_view: ManuallyDrop::new(clear_view),
                    },
                    true,
                );

                let texture = Arc::new(texture);

                device
                    .trackers
                    .lock()
                    .textures
                    .insert_single(&texture, hal::TextureUses::UNINITIALIZED);

                let id = fid.assign(resource::Fallible::Valid(texture));

                if present.acquired_texture.is_some() {
                    return Err(SurfaceError::AlreadyAcquired);
                }
                present.acquired_texture = Some(id);

                let status = if ast.suboptimal {
                    Status::Suboptimal
                } else {
                    Status::Good
                };
                (Some(id), status)
            }
            Ok(None) => (None, Status::Timeout),
            Err(err) => (
                None,
                match err {
                    hal::SurfaceError::Lost => Status::Lost,
                    hal::SurfaceError::Device(err) => {
                        return Err(device.handle_hal_error(err).into());
                    }
                    hal::SurfaceError::Outdated => Status::Outdated,
                    hal::SurfaceError::Other(msg) => {
                        log::error!("acquire error: {}", msg);
                        Status::Lost
                    }
                },
            ),
        };

        Ok(SurfaceOutput { status, texture_id })
    }

    pub fn surface_present(&self, surface_id: id::SurfaceId) -> Result<Status, SurfaceError> {
        profiling::scope!("SwapChain::present");

        let hub = &self.hub;

        let surface = self.surfaces.get(surface_id);

        let mut presentation = surface.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = &present.device;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::Present(surface_id));
        }

        device.check_is_valid()?;
        let queue = device.get_queue().unwrap();

        let result = {
            let texture_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;

            // The texture ID got added to the device tracker by `submit()`,
            // and now we are moving it away.
            let texture = hub.textures.remove(texture_id).get();
            if let Ok(texture) = texture {
                device
                    .trackers
                    .lock()
                    .textures
                    .remove(texture.tracker_index());
                let suf = surface.raw(device.backend()).unwrap();
                let exclusive_snatch_guard = device.snatchable_lock.write();
                match texture.inner.snatch(exclusive_snatch_guard).unwrap() {
                    resource::TextureInner::Surface { raw, parent_id } => {
                        if surface_id != parent_id {
                            log::error!("Presented frame is from a different surface");
                            Err(hal::SurfaceError::Lost)
                        } else {
                            unsafe { queue.raw().present(suf, raw) }
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                Err(hal::SurfaceError::Outdated) //TODO?
            }
        };

        match result {
            Ok(()) => Ok(Status::Good),
            Err(err) => match err {
                hal::SurfaceError::Lost => Ok(Status::Lost),
                hal::SurfaceError::Device(err) => {
                    Err(SurfaceError::from(device.handle_hal_error(err)))
                }
                hal::SurfaceError::Outdated => Ok(Status::Outdated),
                hal::SurfaceError::Other(msg) => {
                    log::error!("acquire error: {}", msg);
                    Err(SurfaceError::Invalid)
                }
            },
        }
    }

    pub fn surface_texture_discard(&self, surface_id: id::SurfaceId) -> Result<(), SurfaceError> {
        profiling::scope!("SwapChain::discard");

        let hub = &self.hub;

        let surface = self.surfaces.get(surface_id);
        let mut presentation = surface.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = &present.device;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::DiscardSurfaceTexture(surface_id));
        }

        device.check_is_valid()?;

        {
            let texture_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;

            // The texture ID got added to the device tracker by `submit()`,
            // and now we are moving it away.
            let texture = hub.textures.remove(texture_id).get();

            if let Ok(texture) = texture {
                device
                    .trackers
                    .lock()
                    .textures
                    .remove(texture.tracker_index());
                let suf = surface.raw(device.backend());
                let exclusive_snatch_guard = device.snatchable_lock.write();
                match texture.inner.snatch(exclusive_snatch_guard).unwrap() {
                    resource::TextureInner::Surface { raw, parent_id } => {
                        if surface_id == parent_id {
                            unsafe { suf.unwrap().discard_texture(raw) };
                        } else {
                            log::warn!("Surface texture is outdated");
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }

        Ok(())
    }
}
