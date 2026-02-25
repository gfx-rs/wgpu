/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use alloc::{sync::Arc, vec, vec::Vec};
use core::mem::ManuallyDrop;
use core::sync::atomic::Ordering;

#[cfg(feature = "trace")]
use crate::device::trace::{Action, IntoTrace};
use crate::{
    conv,
    device::{Device, DeviceError, MissingDownlevelFlags, WaitIdleError},
    global::Global,
    hal_label, id,
    instance::Surface,
    resource, SURFACE_QUEUE_ID,
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
        let e: &dyn WebGpuError = match self {
            Self::Device(e) => e,
            Self::Invalid
            | Self::NotConfigured
            | Self::AlreadyAcquired
            | Self::TextureDestroyed => return ErrorType::Validation,
        };
        e.webgpu_error_type()
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
        let e: &dyn WebGpuError = match self {
            Self::Device(e) => e,
            Self::MissingDownlevelFlags(e) => e,
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
            | Self::UnsupportedUsage { .. } => return ErrorType::Validation,
        };
        e.webgpu_error_type()
    }
}

pub type ResolvedSurfaceOutput = SurfaceOutput<Arc<resource::Texture>>;

#[repr(C)]
#[derive(Debug)]
pub struct SurfaceOutput<T = id::TextureId> {
    pub status: Status,
    pub texture: Option<T>,
}

/// Ensure a surface texture is in `TextureUses::PRESENT` state before
/// presenting it.
///
/// If the texture was used in a normal submission, the submission machinery
/// already transitioned it to `PRESENT`. This function is a no-op in that
/// case.
///
/// If the texture was acquired but never used in any command buffer
/// submission, it is still in `TextureUses::UNINITIALIZED`. This function
/// detects that case and submits a command buffer with the layout transition.
fn ensure_surface_texture_is_presentable(
    device: &Arc<Device>,
    queue: &crate::device::queue::PerQueueData,
    texture: &Arc<resource::Texture>,
    raw_surface_texture: &dyn hal::DynSurfaceTexture,
) -> Result<(), DeviceError> {
    let pending_transitions: Vec<crate::track::PendingTransition<wgt::TextureUses>> = {
        let mut trackers = queue.trackers.lock();
        trackers
            .textures
            .set_single(
                texture,
                texture.full_range.clone(),
                wgt::TextureUses::PRESENT,
            )
            .collect()
    };

    if pending_transitions.is_empty() {
        return Ok(());
    }

    let raw_tex: &dyn hal::DynTexture = raw_surface_texture.borrow();
    let barriers: Vec<_> = pending_transitions
        .into_iter()
        .map(|t| t.into_hal(raw_tex))
        .collect();

    let mut encoder = queue
        .command_allocator
        .acquire_encoder(device.raw(), queue.raw.as_ref())
        .map_err(|e| device.handle_hal_error(e))?;
    unsafe {
        encoder
            .begin_encoding(hal_label(
                Some("(wgpu internal) surface texture present transition"),
                device.instance_flags,
            ))
            .map_err(|e| device.handle_hal_error(e))?;
        encoder.transition_textures(&barriers);
    }
    let cmd_buf = unsafe { encoder.end_encoding() }.map_err(|e| device.handle_hal_error(e))?;

    let submit_index = {
        let mut fence = queue.fence.write();
        let mut cmd_indices = queue.command_indices.write();
        cmd_indices.active_submission_index += 1;
        let submit_index = cmd_indices.active_submission_index;
        drop(cmd_indices);

        unsafe {
            queue
                .raw
                .submit(&mut [hal::QueueSubmitInfo {
                    command_buffers: &[cmd_buf.as_ref()],
                    surface_textures: &[raw_surface_texture],
                    signal_fences: &mut [(fence.as_mut(), submit_index)],
                    wait_fences: &mut [],
                }])
                .map_err(|e| device.handle_hal_error(e))?;
        }

        queue
            .last_successful_submission_index
            .fetch_max(submit_index, Ordering::SeqCst);
        submit_index
    };

    // Wrap the encoder so its Drop impl recycles it back to the allocator.
    let inner_encoder = crate::command::InnerCommandEncoder {
        raw: ManuallyDrop::new(encoder),
        list: vec![cmd_buf],
        device: device.clone(),
        queue_index: SURFACE_QUEUE_ID,
        is_open: false,
        api: crate::command::EncodingApi::InternalUse,
        label: "(wgpu internal) surface texture present transition".into(),
    };

    if let Some(queue_arc) = device.get_queue(SURFACE_QUEUE_ID) {
        queue_arc.track_present_encoder(inner_encoder, submit_index);
    } else {
        // If the queue has been dropped then all work has already completed. The stall should be negligible.
        let fence = queue.fence.read();
        unsafe { device.raw().wait(fence.as_ref(), submit_index, None) }
            .map_err(|e| device.handle_hal_error(e))?;
        drop(fence);
        drop(inner_encoder);
    }

    Ok(())
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

        let queue = device.get_queue_shared(SURFACE_QUEUE_ID);

        let fence = queue.fence.read();

        let suf = self.raw(device.backend()).unwrap();
        let (texture, status) = match unsafe {
            suf.acquire_texture(
                Some(core::time::Duration::from_millis(FRAME_TIMEOUT_MS as u64)),
                fence.as_ref(),
            )
        } {
            Ok(Some(ast)) => {
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
                    initial_queue: SURFACE_QUEUE_ID,
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
                    .get_queue_shared(SURFACE_QUEUE_ID)
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
                        log::error!("acquire error: {msg}");
                        Status::Lost
                    }
                },
            ),
        };

        Ok(ResolvedSurfaceOutput { status, texture })
    }

    pub fn present(&self) -> Result<Status, SurfaceError> {
        profiling::scope!("Surface::present");

        let mut presentation = self.presentation.lock();
        let present = match presentation.as_mut() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        let device = &present.device;

        device.check_is_valid()?;
        let queue = device.get_queue_shared(SURFACE_QUEUE_ID);

        let texture = present
            .acquired_texture
            .take()
            .ok_or(SurfaceError::AlreadyAcquired)?;

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = texture.inner.snatch(&mut exclusive_snatch_guard);
        drop(exclusive_snatch_guard);

        let result = match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = self.raw(device.backend()).unwrap();

                // Transition the texture to PRESENT layout if it has not been
                // used in any command buffer submission.  In the normal path
                // this is a no-op because the submission machinery already
                // performed the transition; it only does real work when the
                // caller acquired the texture but never rendered to it.
                ensure_surface_texture_is_presentable(device, queue, &texture, raw.as_ref())
                    .map_err(SurfaceError::Device)?;

                unsafe { queue.raw.present(raw_surface, raw) }
            }
            _ => unreachable!(),
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
                    log::error!("acquire error: {msg}");
                    Err(SurfaceError::Invalid)
                }
            },
        }
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

    pub fn surface_present(&self, surface_id: id::SurfaceId) -> Result<Status, SurfaceError> {
        let surface = self.surfaces.get(surface_id);

        #[cfg(feature = "trace")]
        if let Some(present) = surface.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                trace.add(Action::Present(surface.to_trace()));
            }
        }

        surface.present()
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
