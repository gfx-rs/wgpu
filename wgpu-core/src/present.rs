/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{mem::ManuallyDrop, sync::atomic::Ordering};

#[cfg(feature = "trace")]
use crate::device::trace::{Action, IntoTrace};
use crate::{
    conv,
    device::{
        queue::Queue, Device, DeviceError, DeviceMismatch, MissingDownlevelFlags, WaitIdleError,
    },
    global::Global,
    hal_label, id,
    instance::Surface,
    resource::{self, Labeled},
    SubmissionIndex,
};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    SurfaceStatus as Status,
};

const FRAME_TIMEOUT_MS: u32 = 1000;

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device: Arc<Device>,
    pub(crate) config: wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
    pub(crate) acquired_texture: Option<Arc<resource::Texture>>,
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
    #[error("Texture has been destroyed")]
    TextureDestroyed,
}

impl WebGpuError for SurfaceError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::Invalid
            | Self::NotConfigured
            | Self::AlreadyAcquired
            | Self::TextureDestroyed => ErrorType::Validation,
        }
    }
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
    #[error("Failed to wait for GPU to come idle before reconfiguring the Surface")]
    GpuWaitTimeout,
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
        requested: wgt::TextureUses,
        available: wgt::TextureUses,
    },
}

impl From<WaitIdleError> for ConfigureSurfaceError {
    fn from(e: WaitIdleError) -> Self {
        match e {
            WaitIdleError::Device(d) => ConfigureSurfaceError::Device(d),
            WaitIdleError::WrongSubmissionIndex(..) => unreachable!(),
            WaitIdleError::Timeout => ConfigureSurfaceError::GpuWaitTimeout,
        }
    }
}

impl WebGpuError for ConfigureSurfaceError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::Device(e) => e.webgpu_error_type(),
            Self::MissingDownlevelFlags(e) => e.webgpu_error_type(),
            Self::InvalidSurface
            | Self::InvalidViewFormat(..)
            | Self::PreviousOutputExists
            | Self::GpuWaitTimeout
            | Self::ZeroArea
            | Self::TooLarge { .. }
            | Self::UnsupportedQueueFamily
            | Self::UnsupportedFormat { .. }
            | Self::UnsupportedPresentMode { .. }
            | Self::UnsupportedAlphaMode { .. }
            | Self::UnsupportedUsage { .. } => ErrorType::Validation,
        }
    }
}

pub type ResolvedSurfaceOutput = SurfaceOutput<Arc<resource::Texture>>;

#[repr(C)]
#[derive(Debug)]
pub struct SurfaceOutput<T = id::TextureId> {
    pub status: Status,
    pub texture: Option<T>,
}

impl Queue {
    /// Present a surface texture and return the submission index that was
    /// active at the time of the present.
    ///
    /// The returned [`SubmissionIndex`] can be passed to
    /// [`Queue::wait_for_present`] to wait for the presentation to complete.
    /// The surface texture is kept alive internally until a subsequent queue
    /// submission (with index strictly greater than the returned value)
    /// finishes, or until the queue is dropped.
    pub fn present(&self, surface: &Surface) -> Result<(Status, SubmissionIndex), SurfaceError> {
        profiling::scope!("Queue::present");

        self.device.check_is_valid()?;

        let mut presentation_lock = surface.presentation.lock();
        let presentation = presentation_lock.as_mut();

        if let Some(presentation) = presentation.as_ref() {
            let same_device = Arc::ptr_eq(&presentation.device, &self.device);

            if !same_device {
                return Err(SurfaceError::Device(DeviceError::DeviceMismatch(Box::new(
                    DeviceMismatch {
                        res: self.error_ident(),
                        res_device: self.device.error_ident(),
                        target: None,
                        target_device: presentation.device.error_ident(),
                    },
                ))));
            }
        }

        let present = match presentation {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let texture = present
            .acquired_texture
            .take()
            .ok_or(SurfaceError::AlreadyAcquired)?;

        let device = &self.device;

        // If the texture was acquired but never rendered to / submitted, clear
        // it and transition it to the PRESENT state before presenting.
        self.prepare_surface_texture_for_present(&texture)?;

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = texture.inner.snatch(&mut exclusive_snatch_guard);
        drop(exclusive_snatch_guard);

        let result = match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = surface.raw(device.backend()).unwrap();
                let _fence_lock = device.fence.write();
                unsafe { self.raw().present(raw_surface, raw) }
            }
            _ => unreachable!(),
        };

        // Assign a fresh submission index to this present so it is distinct
        // from any prior real queue submission.  This means:
        //
        //   - Waiting for the *submission* index (N) will not accidentally
        //     trigger present resolution — the present is at N+1, which is
        //     strictly after N.
        //   - Waiting for the *present* index (N+1) will correctly block
        //     until the GPU is done with the presentation.
        //
        // We also update `last_successful_submission_index` so that the normal
        // `Device::maintain` wait-validity check does not reject a poll for
        // this index.
        let present_index = {
            let mut indices = device.command_indices.write();
            indices.active_submission_index += 1;
            indices.active_submission_index
        };

        match result {
            Ok(()) => {
                device
                    .last_successful_submission_index
                    .store(present_index, Ordering::Release);
                self.lock_life().track_present(present_index, texture);
                Ok((Status::Good, present_index))
            }
            Err(err) => match err {
                hal::SurfaceError::Timeout => {
                    device
                        .last_successful_submission_index
                        .store(present_index, Ordering::Release);
                    self.lock_life().track_present(present_index, texture);
                    Ok((Status::Timeout, present_index))
                }
                hal::SurfaceError::Occluded => {
                    device
                        .last_successful_submission_index
                        .store(present_index, Ordering::Release);
                    self.lock_life().track_present(present_index, texture);
                    Ok((Status::Occluded, present_index))
                }
                hal::SurfaceError::Lost => {
                    device
                        .last_successful_submission_index
                        .store(present_index, Ordering::Release);
                    self.lock_life().track_present(present_index, texture);
                    Ok((Status::Lost, present_index))
                }
                hal::SurfaceError::Device(err) => {
                    Err(SurfaceError::from(device.handle_hal_error(err)))
                }
                hal::SurfaceError::Outdated => {
                    device
                        .last_successful_submission_index
                        .store(present_index, Ordering::Release);
                    self.lock_life().track_present(present_index, texture);
                    Ok((Status::Outdated, present_index))
                }
                hal::SurfaceError::Other(msg) => {
                    log::error!("present error: {msg}");
                    Err(SurfaceError::Invalid)
                }
            },
        }
    }
}

impl Surface {
    pub fn get_current_texture(&self) -> Result<ResolvedSurfaceOutput, SurfaceError> {
        profiling::scope!("Surface::get_current_texture");

        let (device, config) = if let Some(ref present) = *self.presentation.lock() {
            present.device.check_is_valid()?;
            (present.device.clone(), present.config.clone())
        } else {
            return Err(SurfaceError::NotConfigured);
        };

        let fence = device.fence.read();

        let suf = self.raw(device.backend()).unwrap();
        let (texture, status) = match unsafe {
            suf.acquire_texture(
                Some(core::time::Duration::from_millis(FRAME_TIMEOUT_MS as u64)),
                fence.as_ref(),
            )
        } {
            Ok(ast) => {
                drop(fence);

                let texture_desc = wgt::TextureDescriptor {
                    label: hal_label(
                        Some(alloc::borrow::Cow::Borrowed("<Surface Texture>")),
                        device.instance_flags,
                    ),
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
                let format_features = wgt::TextureFormatFeatures {
                    allowed_usages: wgt::TextureUsages::RENDER_ATTACHMENT,
                    flags: wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4
                        | wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
                };
                let hal_usage = conv::map_texture_usage(
                    config.usage,
                    config.format.into(),
                    format_features.flags,
                );
                let clear_view_desc = hal::TextureViewDescriptor {
                    label: hal_label(
                        Some("(wgpu internal) clear surface texture view"),
                        device.instance_flags,
                    ),
                    format: config.format,
                    dimension: wgt::TextureViewDimension::D2,
                    usage: wgt::TextureUses::COLOR_TARGET,
                    range: wgt::ImageSubresourceRange::default(),
                };
                let clear_view = unsafe {
                    device
                        .raw()
                        .create_texture_view(ast.texture.as_ref().borrow(), &clear_view_desc)
                }
                .map_err(|e| device.handle_hal_error(e))?;

                let mut presentation = self.presentation.lock();
                let present = presentation.as_mut().unwrap();
                let texture = resource::Texture::new(
                    &device,
                    resource::TextureInner::Surface { raw: ast.texture },
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
                    .insert_single(&texture, wgt::TextureUses::UNINITIALIZED);

                if present.acquired_texture.is_some() {
                    return Err(SurfaceError::AlreadyAcquired);
                }
                present.acquired_texture = Some(texture.clone());

                let status = if ast.suboptimal {
                    Status::Suboptimal
                } else {
                    Status::Good
                };
                (Some(texture), status)
            }
            Err(err) => (
                None,
                match err {
                    hal::SurfaceError::Timeout => Status::Timeout,
                    hal::SurfaceError::Occluded => Status::Occluded,
                    hal::SurfaceError::Lost => Status::Lost,
                    hal::SurfaceError::Device(err) => {
                        return Err(device.handle_hal_error(err).into());
                    }
                    hal::SurfaceError::Outdated => Status::Outdated,
                    hal::SurfaceError::Other(msg) => {
                        log::error!("acquire error: {msg}");
                        Status::Lost
                    }
                },
            ),
        };

        Ok(ResolvedSurfaceOutput { status, texture })
    }

    pub fn discard(&self) -> Result<(), SurfaceError> {
        profiling::scope!("Surface::discard");

        let mut presentation = self.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = &present.device;

        device.check_is_valid()?;

        let texture = present
            .acquired_texture
            .take()
            .ok_or(SurfaceError::AlreadyAcquired)?;

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = texture.inner.snatch(&mut exclusive_snatch_guard);
        drop(exclusive_snatch_guard);

        match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = self.raw(device.backend()).unwrap();
                unsafe { raw_surface.discard_texture(raw) };
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Global {
    pub fn surface_get_current_texture(
        &self,
        surface_id: id::SurfaceId,
        texture_id_in: Option<id::TextureId>,
    ) -> Result<SurfaceOutput, SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        let fid = self.hub.textures.prepare(texture_id_in);

        let output = surface.get_current_texture()?;

        #[cfg(feature = "trace")]
        if let Some(present) = surface.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                if let Some(texture) = present.acquired_texture.as_ref() {
                    trace.add(Action::GetSurfaceTexture {
                        id: texture.to_trace(),
                        parent: surface.to_trace(),
                    });
                }
            }
        }

        let status = output.status;
        let texture_id = output
            .texture
            .map(|texture| fid.assign(resource::Fallible::Valid(texture)));

        Ok(SurfaceOutput {
            status,
            texture: texture_id,
        })
    }

    pub fn queue_present(
        &self,
        queue_id: id::QueueId,
        surface_id: id::SurfaceId,
    ) -> Result<(Status, SubmissionIndex), SurfaceError> {
        let queue = self.hub.queues.get(queue_id);
        let surface = self.surfaces.get(surface_id);

        let result = queue.present(&surface);

        #[cfg(feature = "trace")]
        if let Ok((_, present_index)) = &result {
            if let Some(presentation) = surface.presentation.lock().as_ref() {
                if let Some(ref mut trace) = *presentation.device.trace.lock() {
                    trace.add(Action::Present(*present_index, surface.to_trace()));
                }
            }
        }

        result
    }

    /// TODO: is this needed by deno?
    pub fn surface_present(&self, surface_id: id::SurfaceId) -> Result<Status, SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        let queue = {
            let lock = surface.presentation.lock();
            let present = lock.as_ref().ok_or(SurfaceError::NotConfigured)?;
            present.device.get_queue().unwrap()
        };

        let result = queue.present(&surface);

        #[cfg(feature = "trace")]
        if let Ok((_, present_index)) = &result {
            if let Some(present) = surface.presentation.lock().as_ref() {
                if let Some(ref mut trace) = *present.device.trace.lock() {
                    trace.add(Action::Present(*present_index, surface.to_trace()));
                }
            }
        }

        result.map(|(status, _)| status)
    }

    pub fn surface_texture_discard(&self, surface_id: id::SurfaceId) -> Result<(), SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        #[cfg(feature = "trace")]
        if let Some(present) = surface.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                trace.add(Action::DiscardSurfaceTexture(surface.to_trace()));
            }
        }

        surface.discard()
    }
}
