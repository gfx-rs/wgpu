/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use std::{
    borrow::Borrow,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    conv,
    device::any_device::AnyDevice,
    device::{DeviceError, MissingDownlevelFlags, WaitIdleError},
    global::Global,
    hal_api::HalApi,
    hal_label,
    id::{SurfaceId, TextureId},
    identity::{GlobalIdentityHandlerFactory, Input},
    init_tracker::TextureInitTracker,
    resource::{self, ResourceInfo},
    snatch::Snatchable,
    track,
};

use hal::{Queue as _, Surface as _};
use parking_lot::RwLock;
use thiserror::Error;
use wgt::SurfaceStatus as Status;

const FRAME_TIMEOUT_MS: u32 = 1000;
pub const DESIRED_NUM_FRAMES: u32 = 3;

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device: AnyDevice,
    pub(crate) config: wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
    #[allow(unused)]
    pub(crate) num_frames: u32,
    pub(crate) acquired_texture: Option<TextureId>,
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
    #[error("`Surface` width and height must be within the maximum supported texture size. Requested was ({width}, {height}), maximum extent is {max_texture_dimension_2d}.")]
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
    #[error("Requested usage is not supported")]
    UnsupportedUsage,
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
    pub texture_id: Option<TextureId>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn surface_get_current_texture<A: HalApi>(
        &self,
        surface_id: SurfaceId,
        texture_id_in: Input<G, TextureId>,
    ) -> Result<SurfaceOutput, SurfaceError> {
        profiling::scope!("SwapChain::get_next_texture");

        let hub = A::hub(self);

        let fid = hub.textures.prepare::<G>(texture_id_in);

        let surface = self
            .surfaces
            .get(surface_id)
            .map_err(|_| SurfaceError::Invalid)?;

        let (device, config) = if let Some(ref present) = *surface.presentation.lock() {
            match present.device.downcast_clone::<A>() {
                Some(device) => {
                    if !device.is_valid() {
                        return Err(DeviceError::Lost.into());
                    }
                    (device, present.config.clone())
                }
                None => return Err(SurfaceError::NotConfigured),
            }
        } else {
            return Err(SurfaceError::NotConfigured);
        };

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::GetSurfaceTexture {
                id: fid.id(),
                parent_id: surface_id,
            });
        }
        #[cfg(not(feature = "trace"))]
        let _ = device;

        let suf = A::get_surface(surface.as_ref());
        let (texture_id, status) = match unsafe {
            suf.unwrap()
                .raw
                .acquire_texture(Some(std::time::Duration::from_millis(
                    FRAME_TIMEOUT_MS as u64,
                )))
        } {
            Ok(Some(ast)) => {
                let texture_desc = wgt::TextureDescriptor {
                    label: (),
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
                        self.instance.flags,
                    ),
                    format: config.format,
                    dimension: wgt::TextureViewDimension::D2,
                    usage: hal::TextureUses::COLOR_TARGET,
                    range: wgt::ImageSubresourceRange::default(),
                };
                let clear_view = unsafe {
                    hal::Device::create_texture_view(
                        device.raw(),
                        ast.texture.borrow(),
                        &clear_view_desc,
                    )
                }
                .map_err(DeviceError::from)?;

                let mut presentation = surface.presentation.lock();
                let present = presentation.as_mut().unwrap();
                let texture = resource::Texture {
                    inner: Snatchable::new(resource::TextureInner::Surface {
                        raw: Some(ast.texture),
                        parent_id: surface_id,
                        has_work: AtomicBool::new(false),
                    }),
                    device: device.clone(),
                    desc: texture_desc,
                    hal_usage,
                    format_features,
                    initialization_status: RwLock::new(TextureInitTracker::new(1, 1)),
                    full_range: track::TextureSelector {
                        layers: 0..1,
                        mips: 0..1,
                    },
                    info: ResourceInfo::new("<Surface>"),
                    clear_mode: RwLock::new(resource::TextureClearMode::Surface {
                        clear_view: Some(clear_view),
                    }),
                };

                let (id, resource) = fid.assign(texture);
                log::debug!("Created CURRENT Surface Texture {:?}", id);

                {
                    // register it in the device tracker as uninitialized
                    let mut trackers = device.trackers.lock();
                    trackers
                        .textures
                        .insert_single(id, resource, hal::TextureUses::UNINITIALIZED);
                }

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

        Ok(SurfaceOutput { status, texture_id })
    }

    pub fn surface_present<A: HalApi>(
        &self,
        surface_id: SurfaceId,
    ) -> Result<Status, SurfaceError> {
        profiling::scope!("SwapChain::present");

        let hub = A::hub(self);

        let surface = self
            .surfaces
            .get(surface_id)
            .map_err(|_| SurfaceError::Invalid)?;

        let mut presentation = surface.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = present.device.downcast_ref::<A>().unwrap();
        if !device.is_valid() {
            return Err(DeviceError::Lost.into());
        }
        let queue_id = device.queue_id.read().unwrap();
        let queue = hub.queues.get(queue_id).unwrap();

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::Present(surface_id));
        }

        let result = {
            let texture_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;

            // The texture ID got added to the device tracker by `submit()`,
            // and now we are moving it away.
            log::debug!(
                "Removing swapchain texture {:?} from the device tracker",
                texture_id
            );
            device.trackers.lock().textures.remove(texture_id);

            let texture = hub.textures.unregister(texture_id);
            if let Some(texture) = texture {
                let mut exclusive_snatch_guard = device.snatchable_lock.write();
                let suf = A::get_surface(&surface);
                let mut inner = texture.inner_mut(&mut exclusive_snatch_guard);
                let inner = inner.as_mut().unwrap();

                match *inner {
                    resource::TextureInner::Surface {
                        ref mut raw,
                        ref parent_id,
                        ref has_work,
                    } => {
                        if surface_id != *parent_id {
                            log::error!("Presented frame is from a different surface");
                            Err(hal::SurfaceError::Lost)
                        } else if !has_work.load(Ordering::Relaxed) {
                            log::error!("No work has been submitted for this frame");
                            unsafe { suf.unwrap().raw.discard_texture(raw.take().unwrap()) };
                            Err(hal::SurfaceError::Outdated)
                        } else {
                            unsafe {
                                queue
                                    .raw
                                    .as_ref()
                                    .unwrap()
                                    .present(&suf.unwrap().raw, raw.take().unwrap())
                            }
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                Err(hal::SurfaceError::Outdated) //TODO?
            }
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

    pub fn surface_texture_discard<A: HalApi>(
        &self,
        surface_id: SurfaceId,
    ) -> Result<(), SurfaceError> {
        profiling::scope!("SwapChain::discard");

        let hub = A::hub(self);

        let surface = self
            .surfaces
            .get(surface_id)
            .map_err(|_| SurfaceError::Invalid)?;
        let mut presentation = surface.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = present.device.downcast_ref::<A>().unwrap();
        if !device.is_valid() {
            return Err(DeviceError::Lost.into());
        }

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            trace.add(Action::DiscardSurfaceTexture(surface_id));
        }

        {
            let texture_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;

            // The texture ID got added to the device tracker by `submit()`,
            // and now we are moving it away.
            log::debug!(
                "Removing swapchain texture {:?} from the device tracker",
                texture_id
            );
            device.trackers.lock().textures.remove(texture_id);

            let texture = hub.textures.unregister(texture_id);
            if let Some(texture) = texture {
                let suf = A::get_surface(&surface);
                let exclusive_snatch_guard = device.snatchable_lock.write();
                match texture.inner.snatch(exclusive_snatch_guard).unwrap() {
                    resource::TextureInner::Surface {
                        mut raw,
                        parent_id,
                        has_work: _,
                    } => {
                        if surface_id == parent_id {
                            unsafe { suf.unwrap().raw.discard_texture(raw.take().unwrap()) };
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
