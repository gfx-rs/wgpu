/*! Presentation.

    ## Lifecycle

    Whenever a submission detects the use of any surface texture, it adds it to the device
    tracker for the duration of the submission (temporarily, while recording).
    It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
    When this texture is presented, we remove it from the device tracker as well as
    extract it from the hub.
!*/

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    conv,
    device::DeviceError,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{DeviceId, SurfaceId, TextureId, Valid},
    resource,
    track::TextureSelector,
    LifeGuard, Stored,
};

use hal::{Queue as _, Surface as _};
use thiserror::Error;
use wgt::SurfaceStatus as Status;

const FRAME_TIMEOUT_MS: u32 = 1000;
pub const DESIRED_NUM_FRAMES: u32 = 3;

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device_id: Stored<DeviceId>,
    pub(crate) config: wgt::SurfaceConfiguration,
    pub(crate) num_frames: u32,
    pub(crate) acquired_texture: Option<Stored<TextureId>>,
}

impl Presentation {
    pub(crate) fn backend(&self) -> wgt::Backend {
        crate::id::TypedId::unzip(self.device_id.value.0).2
    }
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
    pub texture_id: Option<TextureId>,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn surface_get_current_texture<A: HalApi>(
        &self,
        surface_id: SurfaceId,
        texture_id_in: Input<G, TextureId>,
    ) -> Result<SurfaceOutput, SurfaceError> {
        profiling::scope!("get_next_texture", "SwapChain");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.textures.prepare(texture_id_in);

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let surface = surface_guard
            .get_mut(surface_id)
            .map_err(|_| SurfaceError::Invalid)?;
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device, config) = match surface.presentation {
            Some(ref present) => {
                let device = &device_guard[present.device_id.value];
                (device, present.config.clone())
            }
            None => return Err(SurfaceError::NotConfigured),
        };

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(Action::GetSurfaceTexture {
                id: fid.id(),
                parent_id: surface_id,
            });
        }
        #[cfg(not(feature = "trace"))]
        let _ = device;

        let suf = A::get_surface_mut(surface);
        let (texture_id, status) = match unsafe { suf.raw.acquire_texture(FRAME_TIMEOUT_MS) } {
            Ok(Some(ast)) => {
                let present = surface.presentation.as_mut().unwrap();
                let texture = resource::Texture {
                    inner: resource::TextureInner::Surface {
                        raw: ast.texture,
                        parent_id: Valid(surface_id),
                    },
                    device_id: present.device_id.clone(),
                    desc: wgt::TextureDescriptor {
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
                    },
                    hal_usage: conv::map_texture_usage(config.usage, config.format.into()),
                    format_features: wgt::TextureFormatFeatures {
                        allowed_usages: wgt::TextureUsages::RENDER_ATTACHMENT,
                        flags: wgt::TextureFormatFeatureFlags::empty(),
                        filterable: false,
                    },
                    full_range: TextureSelector {
                        layers: 0..1,
                        levels: 0..1,
                    },
                    life_guard: LifeGuard::new("<Surface>"),
                };

                let ref_count = texture.life_guard.add_ref();
                let id = fid.assign(texture, &mut token);

                //suf.acquired_texture = Some(suf_texture);

                if present.acquired_texture.is_some() {
                    return Err(SurfaceError::AlreadyAcquired);
                }
                present.acquired_texture = Some(Stored {
                    value: id,
                    ref_count,
                });

                let status = if ast.suboptimal {
                    Status::Suboptimal
                } else {
                    Status::Good
                };
                (Some(id.0), status)
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
            let texture_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;

            // The texture ID got added to the device tracker by `submit()`,
            // and now we are moving it away.
            device.trackers.lock().textures.remove(texture_id.value);

            let (texture, _) = hub.textures.unregister(texture_id.value.0, &mut token);
            if let Some(texture) = texture {
                let suf_texture = match texture.inner {
                    resource::TextureInner::Surface { raw, .. } => raw,
                    resource::TextureInner::Native { .. } => unreachable!(),
                };
                let suf = A::get_surface_mut(surface);
                unsafe { device.queue.present(&mut suf.raw, suf_texture) }
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
}
