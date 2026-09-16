/*! Presentation.

## Lifecycle

Whenever a submission detects the use of any surface texture, it adds it to the device
tracker for the duration of the submission (temporarily, while recording).
It's added with `UNINITIALIZED` state and transitioned into `empty()` state.
When this texture is presented, we remove it from the device tracker as well as
extract it from the hub.
!*/

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::mem::ManuallyDrop;

#[cfg(feature = "trace")]
use crate::device::trace::{Action, IntoTrace};
use crate::{
    conv,
    device::{queue::Queue, Device, DeviceError, MissingDownlevelFlags, WaitIdleError},
    hal_label,
    init_tracker::TextureInitTracker,
    instance::Surface,
    resource::{self, Labeled, RawResourceAccess},
};

use thiserror::Error;
use wgt::{
    error::{ErrorType, WebGpuError},
    SurfaceStatus as Status,
};

const FRAME_TIMEOUT_MS: u32 = 1000;

/// Label of the texture that wraps the swapchain image.
const SURFACE_TEXTURE_LABEL: &str = "<Surface Texture>";
/// Label of the intermediate texture handed out when surface view formats are
/// emulated.
const INTERMEDIATE_TEXTURE_LABEL: &str = "<Intermediate Surface Texture>";

#[derive(Debug)]
pub(crate) struct Presentation {
    pub(crate) device: Arc<Device>,
    pub(crate) config: wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>>,
    /// The texture handed out by [`Surface::get_current_texture`].
    ///
    /// This is the swapchain image itself, unless surface view formats are
    /// emulated, in which case it is a separate texture that is copied into the
    /// swapchain image when presenting.
    pub(crate) acquired_texture: Option<Arc<resource::Texture>>,
    /// The actual swapchain image, only set when surface view formats are
    /// emulated.
    pub(crate) acquired_surface_texture: Option<Arc<resource::Texture>>,
    /// The intermediate texture handed out by the previous
    /// [`Surface::get_current_texture`], reused for the next one.
    pub(crate) intermediate_texture: Option<Arc<resource::Texture>>,
    /// Whether surface view formats are emulated with an intermediate texture
    /// and a copy at present time.
    pub(crate) emulate_view_formats: bool,
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
    #[error("No surface image is currently acquired to present")]
    NothingToPresent,
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
            | Self::NothingToPresent
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
    #[error("The `SurfaceOutput` returned by `get_current_texture` must be dropped before re-configuring via `configure` or  retrieving a new texture via `get_current_texture`.")]
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
    #[error("Requested color space {requested:?} is not in the list of color spaces supported for format {format:?}: {available:?}")]
    UnsupportedColorSpace {
        requested: wgt::SurfaceColorSpace,
        format: wgt::TextureFormat,
        available: wgt::SurfaceColorSpaces,
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
    #[error(
        "The surface cannot present the requested view formats {view_formats:?}: the backend does \
         not support them on its swapchain images, and the surface does not support `COPY_DST`, \
         which would be needed to emulate them with an intermediate texture"
    )]
    UnsupportedViewFormats {
        view_formats: Vec<wgt::TextureFormat>,
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
            | Self::UnsupportedColorSpace { .. }
            | Self::UnsupportedPresentMode { .. }
            | Self::UnsupportedAlphaMode { .. }
            | Self::UnsupportedUsage { .. }
            | Self::UnsupportedViewFormats { .. } => ErrorType::Validation,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SurfaceOutput<T = Arc<resource::Texture>> {
    pub status: Status,
    pub texture: Option<T>,
}

impl Surface {
    pub fn get_current_texture(self: &Arc<Self>) -> Result<SurfaceOutput, SurfaceError> {
        let output = self.get_current_texture_inner();
        #[cfg(feature = "trace")]
        if let Some(present) = self.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                if let Some(texture) = present.acquired_texture.as_ref() {
                    trace.add(Action::GetSurfaceTexture {
                        id: texture.to_trace(),
                        parent: self.to_trace(),
                    });
                }
            }
        }
        output
    }

    pub(crate) fn get_current_texture_inner(&self) -> Result<SurfaceOutput, SurfaceError> {
        profiling::scope!("Surface::get_current_texture");

        let (device, config, emulate_view_formats) =
            if let Some(ref present) = *self.presentation.lock() {
                present.device.check_is_valid()?;
                (
                    present.device.clone(),
                    present.config.clone(),
                    present.emulate_view_formats,
                )
            } else {
                return Err(SurfaceError::NotConfigured);
            };

        let suf = self.raw(device.backend()).unwrap();
        let (texture, status) = match unsafe {
            suf.acquire_texture(
                Some(core::time::Duration::from_millis(FRAME_TIMEOUT_MS as u64)),
                device.fence.as_ref(),
            )
        } {
            Ok(ast) => {
                let status = if ast.suboptimal {
                    Status::Suboptimal
                } else {
                    Status::Good
                };

                let texture_desc = wgt::TextureDescriptor {
                    label: hal_label(
                        Some(alloc::borrow::Cow::Borrowed(SURFACE_TEXTURE_LABEL)),
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
                    view_formats: config.view_formats.clone(),
                };
                let format_features = wgt::TextureFormatFeatures {
                    allowed_usages: wgt::TextureUsages::RENDER_ATTACHMENT,
                    flags: wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4
                        | wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
                };

                if emulate_view_formats {
                    let mut presentation = self.presentation.lock();
                    let present = presentation.as_mut().unwrap();
                    if present.acquired_texture.is_some() {
                        return Err(SurfaceError::AlreadyAcquired);
                    }

                    let surface_hal_usage = conv::map_texture_usage(
                        config.usage | wgt::TextureUsages::COPY_DST,
                        config.format.into(),
                        format_features.flags,
                    );
                    let surface_texture = Arc::new(resource::Texture::new(
                        &device,
                        resource::TextureInner::Surface { raw: ast.texture },
                        surface_hal_usage,
                        &texture_desc,
                        format_features,
                        resource::TextureClearMode::None,
                        // The whole image is written by the copy done when
                        // presenting, so it never needs a lazy clear.
                        false,
                    ));
                    device
                        .trackers
                        .lock()
                        .textures
                        .insert_single(&surface_texture, wgt::TextureUses::UNINITIALIZED);

                    // Reuse the intermediate texture of the previous frame. The
                    // application has to present or discard a frame before
                    // acquiring the next one, so the previous frame is done with
                    // it; the queue ordering and the texture tracker take care of
                    // synchronizing with the copy that read from it.
                    let cached = {
                        let snatch_guard = device.snatchable_lock.read();
                        present
                            .intermediate_texture
                            .take()
                            .filter(|texture| texture.raw(&snatch_guard).is_some())
                    };

                    let intermediate = match cached {
                        Some(texture) => {
                            // A new frame has nothing drawn into it yet, so it
                            // has to be cleared again when presented without the
                            // application rendering to it.
                            *texture.initialization_status.write() = TextureInitTracker::new(
                                texture_desc.mip_level_count,
                                texture_desc.size.depth_or_array_layers,
                            );
                            texture
                        }
                        None => {
                            // The intermediate is a texture of its own, so it
                            // gets its own label.
                            let mut desc = texture_desc.clone();
                            desc.label = hal_label(
                                Some(alloc::borrow::Cow::Borrowed(INTERMEDIATE_TEXTURE_LABEL)),
                                device.instance_flags,
                            );

                            // `COPY_SRC` is not part of the descriptor the
                            // application sees, but the texture has to be created
                            // with it on the backend so that it can be copied into
                            // the swapchain image.
                            device
                                .create_texture_with_extra_hal_usage(
                                    &desc,
                                    wgt::TextureUses::COPY_SRC,
                                )
                                .map_err(|error| {
                                    log::error!("failed to create the intermediate texture for surface: {error}");
                                    match error {
                                        resource::CreateTextureError::Device(error) => {
                                            SurfaceError::Device(error)
                                        }
                                        _ => SurfaceError::Device(DeviceError::Lost),
                                    }
                                })?
                        }
                    };
                    present.intermediate_texture = Some(intermediate.clone());

                    present.acquired_surface_texture = Some(surface_texture);
                    present.acquired_texture = Some(intermediate.clone());

                    (Some(intermediate), status)
                } else {
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
                        swizzle: wgt::TextureComponentSwizzle::default(),
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

                    (Some(texture), status)
                }
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

        Ok(SurfaceOutput { status, texture })
    }

    pub fn present(self: &Arc<Self>) -> Result<Status, SurfaceError> {
        #[cfg(feature = "trace")]
        if let Some(present) = self.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                trace.add(Action::Present(self.to_trace()));
            }
        }
        self.present_inner()
    }

    pub(crate) fn present_inner(&self) -> Result<Status, SurfaceError> {
        profiling::scope!("Surface::present");

        let presentation = self.presentation.lock();
        let present = match presentation.as_ref() {
            Some(present) => present,
            None => return Err(SurfaceError::NotConfigured),
        };

        present.device.check_is_valid()?;
        let queue = present
            .device
            .get_queue()
            .ok_or(SurfaceError::Device(DeviceError::Lost))?;
        drop(presentation);

        queue.present(self)
    }
}

impl Queue {
    pub fn present(&self, surface: &Surface) -> Result<Status, SurfaceError> {
        profiling::scope!("Queue::present");

        let (texture, surface_texture, emulate_view_formats) = {
            let mut presentation = surface.presentation.lock();
            let present = match presentation.as_mut() {
                Some(present) => present,
                None => return Err(SurfaceError::NotConfigured),
            };

            let device = &self.device;

            // Check the surface is configured for this device.
            if !Arc::ptr_eq(&present.device, device) {
                return Err(SurfaceError::Device(DeviceError::DeviceMismatch(Box::new(
                    crate::device::DeviceMismatch {
                        res: self.error_ident(),
                        res_device: device.error_ident(),
                        target: None,
                        target_device: present.device.error_ident(),
                    },
                ))));
            }

            (
                present
                    .acquired_texture
                    .take()
                    .ok_or(SurfaceError::NothingToPresent)?,
                present.acquired_surface_texture.take(),
                present.emulate_view_formats,
            )
        };

        let device = &self.device;

        let presented_texture =
            self.prepare_present_texture(texture, surface_texture, emulate_view_formats)?;

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = presented_texture
            .state()
            .ok()
            .and_then(|state| state.inner.snatch(&mut exclusive_snatch_guard));
        drop(exclusive_snatch_guard);

        let result = match inner {
            None => return Err(SurfaceError::TextureDestroyed),
            Some(resource::TextureInner::Surface { raw }) => {
                let raw_surface = surface.raw(device.backend()).unwrap();
                let raw_queue = self.raw();
                // [`wgpu_hal::Queue::present`] requires the queue to be synchronized with submit calls and
                // other present calls. Locking command indices prevents submits which must increment the
                // submission index, and by `write`ing prevents other present calls.
                let _command_indices = device.command_indices.write();
                unsafe { raw_queue.present(raw_surface, raw) }
            }
            _ => unreachable!(),
        };

        match result {
            Ok(()) => Ok(Status::Good),
            Err(err) => match err {
                hal::SurfaceError::Timeout => Ok(Status::Timeout),
                hal::SurfaceError::Occluded => Ok(Status::Occluded),
                hal::SurfaceError::Lost => Ok(Status::Lost),
                hal::SurfaceError::Device(err) => {
                    Err(SurfaceError::from(device.handle_hal_error(err)))
                }
                hal::SurfaceError::Outdated => Ok(Status::Outdated),
                hal::SurfaceError::Other(msg) => {
                    log::error!("present error: {msg}");
                    Err(SurfaceError::Invalid)
                }
            },
        }
    }

    /// Gets `texture` ready to be handed to the backend's present call.
    ///
    /// When surface view formats are emulated, `texture` is copied into
    /// `surface_texture` and the latter is returned.
    fn prepare_present_texture(
        &self,
        texture: Arc<resource::Texture>,
        surface_texture: Option<Arc<resource::Texture>>,
        emulate_view_formats: bool,
    ) -> Result<Arc<resource::Texture>, SurfaceError> {
        if emulate_view_formats {
            let surface_texture = surface_texture.ok_or(SurfaceError::NothingToPresent)?;

            // The application may never have rendered to the texture, in which
            // case it is presented as transparent black; either way it is copied
            // into the swapchain image, which is never the application's render
            // target in this path. The submission that does the copy also
            // transitions the swapchain image to PRESENT.
            // Fixes <https://github.com/gfx-rs/wgpu/issues/6748>
            self.prepare_surface_texture_copy_for_present(&texture, &surface_texture)?;

            Ok(surface_texture)
        } else {
            // If the texture was never rendered to, clear it and transition to
            // PRESENT state before presenting.
            // Fixes <https://github.com/gfx-rs/wgpu/issues/6748>
            self.prepare_surface_texture_for_present(&texture)?;

            Ok(texture)
        }
    }
}

impl Surface {
    pub fn discard(self: &Arc<Self>) -> Result<(), SurfaceError> {
        #[cfg(feature = "trace")]
        if let Some(present) = self.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                trace.add(Action::DiscardSurfaceTexture(self.to_trace()));
            }
        }
        self.discard_inner()
    }

    pub(crate) fn discard_inner(&self) -> Result<(), SurfaceError> {
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
            .ok_or(SurfaceError::NothingToPresent)?;
        let presented_texture = present.acquired_surface_texture.take().unwrap_or(texture);

        let mut exclusive_snatch_guard = device.snatchable_lock.write();
        let inner = presented_texture
            .state()
            .ok()
            .and_then(|state| state.inner.snatch(&mut exclusive_snatch_guard));
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

    pub fn release(self: &Arc<Self>) -> Result<(), SurfaceError> {
        #[cfg(feature = "trace")]
        if let Some(present) = self.presentation.lock().as_ref() {
            if let Some(ref mut trace) = *present.device.trace.lock() {
                trace.add(Action::ReleaseSurfaceTexture(self.to_trace()));
            }
        }
        self.release_inner()
    }

    /// Like `discard`, drops the inner texture reference, but skips the
    /// HAL `discard_texture` call. Safe to call during unwinding
    pub(crate) fn release_inner(&self) -> Result<(), SurfaceError> {
        profiling::scope!("Surface::release");

        let mut presentation = self.presentation.lock();
        let Some(present) = presentation.as_mut() else {
            return Err(SurfaceError::NotConfigured);
        };

        // `texture` is dropped here, decrementing the refcount of
        // Arc<SwapchainAcquireSemaphore>. If this was the last Arc, the Texture
        // is freed, which drops NativeSurfaceTextureMetadata and
        // its Arc<SwapchainAcquireSemaphore>.
        _ = present
            .acquired_texture
            .take()
            .ok_or(SurfaceError::NothingToPresent)?;
        _ = present.acquired_surface_texture.take();

        Ok(())
    }
}

#[cfg(all(test, feature = "noop", feature = "wgsl"))]
mod tests {
    use super::*;
    use crate::hal;
    use alloc::vec;
    use core::any::Any;
    use core::sync::atomic::{AtomicBool, Ordering};

    const FORMAT: wgt::TextureFormat = wgt::TextureFormat::Rgba8UnormSrgb;
    const VIEW_FORMAT: wgt::TextureFormat = wgt::TextureFormat::Rgba8Unorm;

    #[derive(Default)]
    struct RecordedSurfaceConfig {
        view_formats: AtomicBool,
        copy_dst: AtomicBool,
    }

    struct MockHalSurface {
        recorded: Arc<RecordedSurfaceConfig>,
    }

    impl hal::DynResource for MockHalSurface {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl hal::Surface for MockHalSurface {
        type A = hal::noop::Api;

        unsafe fn configure(
            &self,
            _device: &hal::noop::Context,
            config: &hal::SurfaceConfiguration,
        ) -> Result<(), hal::SurfaceError> {
            self.recorded
                .view_formats
                .store(!config.view_formats.is_empty(), Ordering::Relaxed);
            self.recorded.copy_dst.store(
                config.usage.contains(wgt::TextureUses::COPY_DST),
                Ordering::Relaxed,
            );
            Ok(())
        }

        unsafe fn unconfigure(&self, _device: &hal::noop::Context) {}

        unsafe fn acquire_texture(
            &self,
            _timeout: Option<core::time::Duration>,
            _fence: &hal::noop::Fence,
        ) -> Result<hal::AcquiredSurfaceTexture<hal::noop::Api>, hal::SurfaceError> {
            Ok(hal::AcquiredSurfaceTexture {
                texture: hal::noop::Resource,
                suboptimal: false,
            })
        }

        unsafe fn discard_texture(&self, _texture: hal::noop::Resource) {}
    }

    fn mock_surface_capabilities(
        native_view_formats: bool,
        copy_dst_supported: bool,
    ) -> hal::SurfaceCapabilities {
        let mut usage = wgt::TextureUses::COLOR_TARGET | wgt::TextureUses::COPY_SRC;
        usage.set(wgt::TextureUses::COPY_DST, copy_dst_supported);

        hal::SurfaceCapabilities {
            formats: vec![wgt::SurfaceFormatCapabilities {
                format: FORMAT,
                color_spaces: wgt::SurfaceColorSpaces::SRGB,
            }],
            maximum_frame_latency: 1..=4,
            current_extent: None,
            usage,
            present_modes: vec![wgt::PresentMode::Fifo],
            composite_alpha_modes: vec![wgt::CompositeAlphaMode::Opaque],
            native_view_formats,
        }
    }

    fn mock_surface_config(
        view_formats: Vec<wgt::TextureFormat>,
    ) -> wgt::SurfaceConfiguration<Vec<wgt::TextureFormat>> {
        wgt::SurfaceConfiguration {
            usage: wgt::TextureUsages::RENDER_ATTACHMENT,
            format: FORMAT,
            color_space: wgt::SurfaceColorSpace::Auto,
            width: 4,
            height: 4,
            present_mode: wgt::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgt::CompositeAlphaMode::Opaque,
            view_formats,
        }
    }

    fn new_noop_device_and_queue() -> (Arc<Device>, Arc<Queue>) {
        let instance = crate::instance::Instance::new(
            "surface-view-formats",
            wgt::InstanceDescriptor {
                backends: wgt::Backends::NOOP,
                backend_options: wgt::BackendOptions {
                    noop: wgt::NoopBackendOptions::enabled(),
                    ..Default::default()
                },
                ..wgt::InstanceDescriptor::new_without_display_handle()
            },
            None,
        );
        let adapter = instance
            .enumerate_adapters(wgt::Backends::NOOP, false)
            .into_iter()
            .next()
            .expect("the noop backend has no adapter");
        adapter
            .request_device(&crate::device::DeviceDescriptor::default())
            .expect("failed to create the noop device")
    }

    fn new_surface_with_mock_hal_surface(recorded: Arc<RecordedSurfaceConfig>) -> Arc<Surface> {
        let mock: Box<dyn hal::DynSurface> = Box::new(MockHalSurface { recorded });
        Arc::new(Surface {
            presentation: crate::lock::Mutex::new(crate::lock::rank::SURFACE_PRESENTATION, None),
            surface_per_backend: core::iter::once((wgt::Backend::Noop, mock)).collect(),
        })
    }

    #[test]
    fn surface_view_formats_fast_path() {
        // The swapchain image can expose the differing view format itself.
        check_surface_view_formats(true, vec![VIEW_FORMAT], false);
    }

    #[test]
    fn surface_view_formats_polyfill() {
        // The swapchain image can't expose the differing view format, so a
        // separate texture is copied into it when presenting.
        check_surface_view_formats(false, vec![VIEW_FORMAT], true);
    }

    #[test]
    fn surface_view_formats_same_format_is_not_emulated() {
        // A view format equal to the surface format is always allowed, so it
        // never needs a separate texture.
        check_surface_view_formats(false, vec![FORMAT], false);
    }

    #[test]
    fn surface_view_formats_without_copy_dst_is_rejected() {
        // Emulating the differing view format requires copying into the
        // swapchain image, which a surface without `COPY_DST` can't do.
        let (device, _queue) = new_noop_device_and_queue();
        let surface = new_surface_with_mock_hal_surface(Arc::new(RecordedSurfaceConfig::default()));
        let config = mock_surface_config(vec![VIEW_FORMAT]);

        let error = surface
            .configure_with_caps(&device, &config, mock_surface_capabilities(false, false))
            .expect_err("configuring emulated view formats without `COPY_DST` should fail");

        assert!(
            matches!(
                error,
                ConfigureSurfaceError::UnsupportedViewFormats { ref view_formats }
                    if *view_formats == vec![VIEW_FORMAT]
            ),
            "unexpected error: {error:?}"
        );
    }

    /// Checks one combination of surface capabilities and requested view
    /// formats, where `emulate` is whether a separate texture is expected.
    fn check_surface_view_formats(
        native_view_formats: bool,
        view_formats: Vec<wgt::TextureFormat>,
        emulate: bool,
    ) {
        let (device, queue) = new_noop_device_and_queue();
        let recorded = Arc::new(RecordedSurfaceConfig::default());
        let surface = new_surface_with_mock_hal_surface(recorded.clone());
        let config = mock_surface_config(view_formats);
        let has_differing_view_format = config
            .view_formats
            .iter()
            .any(|format| *format != config.format);

        surface
            .configure_with_caps(
                &device,
                &config,
                mock_surface_capabilities(native_view_formats, true),
            )
            .expect("failed to configure the surface");

        // The fast path hands the differing view formats to the swapchain, the
        // polyfill keeps them off the swapchain and makes it a copy destination.
        assert_eq!(
            recorded.view_formats.load(Ordering::Relaxed),
            has_differing_view_format && !emulate
        );
        assert_eq!(recorded.copy_dst.load(Ordering::Relaxed), emulate);

        let output = surface
            .get_current_texture()
            .expect("failed to acquire a surface texture");
        let texture = output.texture.expect("no surface texture");

        // Whatever path is taken, the application sees a texture with exactly
        // the properties it configured.
        assert_eq!(texture.desc.format, FORMAT);
        assert_eq!(texture.desc.usage, config.usage);
        assert_eq!(texture.desc.view_formats, config.view_formats);
        assert_eq!(
            texture.desc.label,
            if emulate {
                INTERMEDIATE_TEXTURE_LABEL
            } else {
                SURFACE_TEXTURE_LABEL
            },
            "the handed out texture has an unexpected label"
        );

        let surface_texture = {
            let presentation = surface.presentation.lock();
            presentation
                .as_ref()
                .unwrap()
                .acquired_surface_texture
                .clone()
        };
        assert_eq!(surface_texture.is_some(), emulate);

        let handed_out_is_surface = {
            let snatch_guard = device.snatchable_lock.read();
            matches!(
                texture.try_inner(&snatch_guard).unwrap(),
                resource::TextureInner::Surface { .. }
            )
        };
        assert_eq!(handed_out_is_surface, !emulate);

        // The application never rendered to the texture, so presenting has to
        // clear it (to transparent black, see #6748).
        let cleared_texture = texture.clone();
        let presented = queue
            .prepare_present_texture(texture, surface_texture, emulate)
            .expect("failed to prepare the texture for presentation");

        let initialized = cleared_texture
            .initialization_status
            .read()
            .mips
            .first()
            .is_some_and(|mip| mip.check(0..1).is_none());
        assert!(initialized, "the texture was not cleared before presenting");

        let presented_is_surface = {
            let snatch_guard = device.snatchable_lock.read();
            matches!(
                presented
                    .state()
                    .ok()
                    .and_then(|state| state.inner.get(&snatch_guard)),
                Some(resource::TextureInner::Surface { .. })
            )
        };
        assert!(presented_is_surface);
    }

    /// Configures `surface` to emulate the view format, so that its frames come
    /// from an intermediate texture.
    fn configure_with_intermediate_texture(surface: &Arc<Surface>, device: &Arc<Device>) {
        surface
            .configure_with_caps(
                device,
                &mock_surface_config(vec![VIEW_FORMAT]),
                mock_surface_capabilities(false, true),
            )
            .expect("failed to configure the surface");
    }

    /// Presents the current frame like [`Queue::present`] does, without a real
    /// backend present call, and returns the texture that was handed out.
    fn present_frame(surface: &Arc<Surface>, queue: &Arc<Queue>) -> Arc<resource::Texture> {
        let output = surface
            .get_current_texture()
            .expect("failed to acquire a surface texture");
        let texture = output.texture.expect("no surface texture");

        let (acquired, surface_texture, emulate_view_formats) = {
            let mut presentation = surface.presentation.lock();
            let present = presentation.as_mut().unwrap();
            (
                present.acquired_texture.take().unwrap(),
                present.acquired_surface_texture.take(),
                present.emulate_view_formats,
            )
        };
        queue
            .prepare_present_texture(acquired, surface_texture, emulate_view_formats)
            .expect("failed to prepare the texture for presentation");

        texture
    }

    #[test]
    fn surface_view_formats_reuses_intermediate_texture() {
        let (device, queue) = new_noop_device_and_queue();
        let surface = new_surface_with_mock_hal_surface(Arc::new(RecordedSurfaceConfig::default()));
        configure_with_intermediate_texture(&surface, &device);

        let first = present_frame(&surface, &queue);
        let second = present_frame(&surface, &queue);

        assert!(
            Arc::ptr_eq(&first, &second),
            "the intermediate texture was not reused"
        );
    }

    #[test]
    fn surface_view_formats_reused_intermediate_texture_is_uninitialized() {
        let (device, queue) = new_noop_device_and_queue();
        let surface = new_surface_with_mock_hal_surface(Arc::new(RecordedSurfaceConfig::default()));
        configure_with_intermediate_texture(&surface, &device);

        // Presenting the first frame clears the texture.
        let first = present_frame(&surface, &queue);
        assert!(
            first
                .initialization_status
                .read()
                .mips
                .first()
                .is_some_and(|mip| mip.check(0..1).is_none()),
            "the first frame was not cleared before presenting"
        );

        // The next frame starts out with nothing drawn into it, so presenting it
        // without the application rendering to it has to clear it again.
        let second = surface
            .get_current_texture()
            .expect("failed to acquire a surface texture")
            .texture
            .expect("no surface texture");
        assert!(
            second
                .initialization_status
                .read()
                .mips
                .first()
                .is_some_and(|mip| mip.check(0..1).is_some()),
            "the reused texture was not reset to uninitialized"
        );
    }

    #[test]
    fn surface_view_formats_destroyed_intermediate_texture_is_not_reused() {
        let (device, queue) = new_noop_device_and_queue();
        let surface = new_surface_with_mock_hal_surface(Arc::new(RecordedSurfaceConfig::default()));
        configure_with_intermediate_texture(&surface, &device);

        let first = present_frame(&surface, &queue);
        first.destroy();

        let second = present_frame(&surface, &queue);
        assert!(
            !Arc::ptr_eq(&first, &second),
            "a destroyed intermediate texture was reused"
        );
    }

    #[test]
    fn surface_view_formats_reconfigure_does_not_reuse_intermediate_texture() {
        let (device, queue) = new_noop_device_and_queue();
        let surface = new_surface_with_mock_hal_surface(Arc::new(RecordedSurfaceConfig::default()));
        configure_with_intermediate_texture(&surface, &device);

        let first = present_frame(&surface, &queue);

        let mut resized_config = mock_surface_config(vec![VIEW_FORMAT]);
        resized_config.width = 8;
        resized_config.height = 8;
        surface
            .configure_with_caps(
                &device,
                &resized_config,
                mock_surface_capabilities(false, true),
            )
            .expect("failed to reconfigure the surface");

        let second = present_frame(&surface, &queue);
        assert!(
            !Arc::ptr_eq(&first, &second),
            "the intermediate texture of a previous configuration was reused"
        );
        assert_eq!(second.desc.size.width, 8);
        assert_eq!(second.desc.size.height, 8);
    }
}
