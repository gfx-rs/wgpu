/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use std::borrow::Borrow;

#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    conv,
    device::DeviceError,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{DeviceId, SurfaceId, TextureId, Valid},
    init_tracker::TextureInitTracker,
    resource, track, LifeGuard, Stored,
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
    #[allow(unused)]
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
        let (texture_id, status) = match unsafe {
            suf.raw
                .acquire_texture(Some(std::time::Duration::from_millis(
                    FRAME_TIMEOUT_MS as u64,
                )))
        } {
            Ok(Some(ast)) => {
                let clear_view_desc = hal::TextureViewDescriptor {
                    label: Some("(wgpu internal) clear surface texture view"),
                    format: config.format,
                    dimension: wgt::TextureViewDimension::D2,
                    usage: hal::TextureUses::COLOR_TARGET,
                    range: wgt::ImageSubresourceRange::default(),
                };
                let mut clear_views = smallvec::SmallVec::new();
                clear_views.push(
                    unsafe {
                        hal::Device::create_texture_view(
                            &device.raw,
                            ast.texture.borrow(),
                            &clear_view_desc,
                        )
                    }
                    .map_err(DeviceError::from)?,
                );

                let present = surface.presentation.as_mut().unwrap();
                let texture = resource::Texture {
                    inner: resource::TextureInner::Surface {
                        raw: ast.texture,
                        parent_id: Valid(surface_id),
                        has_work: false,
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
                        flags: wgt::TextureFormatFeatureFlags::MULTISAMPLE
                            | wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
                    },
                    initialization_status: TextureInitTracker::new(1, 1),
                    full_range: track::TextureSelector {
                        layers: 0..1,
                        mips: 0..1,
                    },
                    life_guard: LifeGuard::new("<Surface>"),
                    clear_mode: resource::TextureClearMode::RenderPass {
                        clear_views,
                        is_color: true,
                    },
                };

                let ref_count = texture.life_guard.add_ref();
                let id = fid.assign(texture, &mut token);

                {
                    // register it in the device tracker as uninitialized
                    let mut trackers = device.trackers.lock();
                    trackers.textures.insert_single(
                        id.0,
                        ref_count.clone(),
                        hal::TextureUses::UNINITIALIZED,
                    );
                }

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
            log::debug!(
                "Removing swapchain texture {:?} from the device tracker",
                texture_id.value
            );
            device.trackers.lock().textures.remove(texture_id.value);

            let (texture, _) = hub.textures.unregister(texture_id.value.0, &mut token);
            if let Some(texture) = texture {
                if let resource::TextureClearMode::RenderPass { clear_views, .. } =
                    texture.clear_mode
                {
                    for clear_view in clear_views {
                        unsafe {
                            hal::Device::destroy_texture_view(&device.raw, clear_view);
                        }
                    }
                }

                let suf = A::get_surface_mut(surface);
                match texture.inner {
                    resource::TextureInner::Surface {
                        raw,
                        parent_id,
                        has_work,
                    } => {
                        if surface_id != parent_id.0 {
                            log::error!("Presented frame is from a different surface");
                            Err(hal::SurfaceError::Lost)
                        } else if !has_work {
                            log::error!("No work has been submitted for this frame");
                            unsafe { suf.raw.discard_texture(raw) };
                            Err(hal::SurfaceError::Outdated)
                        } else {
                            unsafe { device.queue.present(&mut suf.raw, raw) }
                        }
                    }
                    resource::TextureInner::Native { .. } => unreachable!(),
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
        profiling::scope!("discard", "SwapChain");

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
            trace.lock().add(Action::DiscardSurfaceTexture(surface_id));
        }

        {
            let texture_id = present
                .acquired_texture
                .take()
                .ok_or(SurfaceError::AlreadyAcquired)?;

            // The texture ID got added to the device tracker by `submit()`,
            // and now we are moving it away.
            device.trackers.lock().textures.remove(texture_id.value);

            let (texture, _) = hub.textures.unregister(texture_id.value.0, &mut token);
            if let Some(texture) = texture {
                let suf = A::get_surface_mut(surface);
                match texture.inner {
                    resource::TextureInner::Surface {
                        raw,
                        parent_id,
                        has_work: _,
                    } => {
                        if surface_id == parent_id.0 {
                            unsafe { suf.raw.discard_texture(raw) };
                        } else {
                            log::warn!("Surface texture is outdated");
                        }
                    }
                    resource::TextureInner::Native { .. } => unreachable!(),
                }
            }
        }

        Ok(())
    }
}
