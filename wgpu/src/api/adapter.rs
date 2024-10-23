use std::{future::Future, sync::Arc, thread};

use crate::context::{DeviceRequest, DynContext};
use crate::*;

/// Handle to a physical graphics and/or compute device.
///
/// Adapters can be created using [`Instance::request_adapter`]
/// or other [`Instance`] methods.
///
/// Adapters can be used to open a connection to the corresponding [`Device`]
/// on the host system by using [`Adapter::request_device`].
///
/// Does not have to be kept alive.
///
/// Corresponds to [WebGPU `GPUAdapter`](https://gpuweb.github.io/gpuweb/#gpu-adapter).
#[derive(Debug)]
pub struct Adapter {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Adapter: Send, Sync);

impl Drop for Adapter {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.adapter_drop(self.data.as_ref())
        }
    }
}

pub use wgt::RequestAdapterOptions as RequestAdapterOptionsBase;
/// Additional information required when requesting an adapter.
///
/// For use with [`Instance::request_adapter`].
///
/// Corresponds to [WebGPU `GPURequestAdapterOptions`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpurequestadapteroptions).
pub type RequestAdapterOptions<'a, 'b> = RequestAdapterOptionsBase<&'a Surface<'b>>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(RequestAdapterOptions<'_, '_>: Send, Sync);

impl Adapter {
    /// Requests a connection to a physical device, creating a logical device.
    ///
    /// Returns the [`Device`] together with a [`Queue`] that executes command buffers.
    ///
    /// [Per the WebGPU specification], an [`Adapter`] may only be used once to create a device.
    /// If another device is wanted, call [`Instance::request_adapter()`] again to get a fresh
    /// [`Adapter`].
    /// However, `wgpu` does not currently enforce this restriction.
    ///
    /// # Arguments
    ///
    /// - `desc` - Description of the features and limits requested from the given device.
    /// - `trace_path` - Can be used for API call tracing, if that feature is
    ///   enabled in `wgpu-core`.
    ///
    /// # Panics
    ///
    /// - `request_device()` was already called on this `Adapter`.
    /// - Features specified by `desc` are not supported by this adapter.
    /// - Unsafe features were requested but not enabled when requesting the adapter.
    /// - Limits requested exceed the values provided by the adapter.
    /// - Adapter does not support all features wgpu requires to safely operate.
    ///
    /// [Per the WebGPU specification]: https://www.w3.org/TR/webgpu/#dom-gpuadapter-requestdevice
    pub fn request_device(
        &self,
        desc: &DeviceDescriptor<'_>,
        trace_path: Option<&std::path::Path>,
    ) -> impl Future<Output = Result<(Device, Queue), RequestDeviceError>> + WasmNotSend {
        let context = Arc::clone(&self.context);
        let device = DynContext::adapter_request_device(
            &*self.context,
            self.data.as_ref(),
            desc,
            trace_path,
        );
        async move {
            device.await.map(
                |DeviceRequest {
                     device_data,
                     queue_data,
                 }| {
                    (
                        Device {
                            context: Arc::clone(&context),
                            data: device_data,
                        },
                        Queue {
                            context,
                            data: queue_data,
                        },
                    )
                },
            )
        }
    }

    /// Create a wgpu [`Device`] and [`Queue`] from a wgpu-hal `OpenDevice`
    ///
    /// # Safety
    ///
    /// - `hal_device` must be created from this adapter internal handle.
    /// - `desc.features` must be a subset of `hal_device` features.
    #[cfg(wgpu_core)]
    pub unsafe fn create_device_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_device: hal::OpenDevice<A>,
        desc: &DeviceDescriptor<'_>,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Device, Queue), RequestDeviceError> {
        let context = Arc::clone(&self.context);
        unsafe {
            self.context
                .as_any()
                .downcast_ref::<crate::backend::ContextWgpuCore>()
                // Part of the safety requirements is that the device was generated from the same adapter.
                // Therefore, unwrap is fine here since only WgpuCoreContext based adapters have the ability to create hal devices.
                .unwrap()
                .create_device_from_hal(
                    crate::context::downcast_ref(self.data.as_ref()),
                    hal_device,
                    desc,
                    trace_path,
                )
        }
        .map(|(device, queue)| {
            (
                Device {
                    context: Arc::clone(&context),
                    data: Box::new(device),
                },
                Queue {
                    context,
                    data: Box::new(queue),
                },
            )
        })
    }

    /// Apply a callback to this `Adapter`'s underlying backend adapter.
    ///
    /// If this `Adapter` is implemented by the backend API given by `A` (Vulkan,
    /// Dx12, etc.), then apply `hal_adapter_callback` to `Some(&adapter)`, where
    /// `adapter` is the underlying backend adapter type, [`A::Adapter`].
    ///
    /// If this `Adapter` uses a different backend, apply `hal_adapter_callback`
    /// to `None`.
    ///
    /// The adapter is locked for reading while `hal_adapter_callback` runs. If
    /// the callback attempts to perform any `wgpu` operations that require
    /// write access to the adapter, deadlock will occur. The locks are
    /// automatically released when the callback returns.
    ///
    /// # Safety
    ///
    /// - The raw handle passed to the callback must not be manually destroyed.
    ///
    /// [`A::Adapter`]: hal::Api::Adapter
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        hal_adapter_callback: F,
    ) -> R {
        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe {
                ctx.adapter_as_hal::<A, F, R>(
                    crate::context::downcast_ref(self.data.as_ref()),
                    hal_adapter_callback,
                )
            }
        } else {
            hal_adapter_callback(None)
        }
    }

    /// Returns whether this adapter may present to the passed surface.
    pub fn is_surface_supported(&self, surface: &Surface<'_>) -> bool {
        DynContext::adapter_is_surface_supported(
            &*self.context,
            self.data.as_ref(),
            surface.surface_data.as_ref(),
        )
    }

    /// The features which can be used to create devices on this adapter.
    pub fn features(&self) -> Features {
        DynContext::adapter_features(&*self.context, self.data.as_ref())
    }

    /// The best limits which can be used to create devices on this adapter.
    pub fn limits(&self) -> Limits {
        DynContext::adapter_limits(&*self.context, self.data.as_ref())
    }

    /// Get info about the adapter itself.
    pub fn get_info(&self) -> AdapterInfo {
        DynContext::adapter_get_info(&*self.context, self.data.as_ref())
    }

    /// Get info about the adapter itself.
    pub fn get_downlevel_capabilities(&self) -> DownlevelCapabilities {
        DynContext::adapter_downlevel_capabilities(&*self.context, self.data.as_ref())
    }

    /// Returns the features supported for a given texture format by this adapter.
    ///
    /// Note that the WebGPU spec further restricts the available usages/features.
    /// To disable these restrictions on a device, request the [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] feature.
    pub fn get_texture_format_features(&self, format: TextureFormat) -> TextureFormatFeatures {
        DynContext::adapter_get_texture_format_features(&*self.context, self.data.as_ref(), format)
    }

    /// Generates a timestamp using the clock used by the presentation engine.
    ///
    /// When comparing completely opaque timestamp systems, we need a way of generating timestamps that signal
    /// the exact same time. You can do this by calling your own timestamp function immediately after a call to
    /// this function. This should result in timestamps that are 0.5 to 5 microseconds apart. There are locks
    /// that must be taken during the call, so don't call your function before.
    ///
    /// ```no_run
    /// # let adapter: wgpu::Adapter = panic!();
    /// # let some_code = || wgpu::PresentationTimestamp::INVALID_TIMESTAMP;
    /// use std::time::{Duration, Instant};
    /// let presentation = adapter.get_presentation_timestamp();
    /// let instant = Instant::now();
    ///
    /// // We can now turn a new presentation timestamp into an Instant.
    /// let some_pres_timestamp = some_code();
    /// let duration = Duration::from_nanos((some_pres_timestamp.0 - presentation.0) as u64);
    /// let new_instant: Instant = instant + duration;
    /// ```
    //
    /// [Instant]: std::time::Instant
    pub fn get_presentation_timestamp(&self) -> PresentationTimestamp {
        DynContext::adapter_get_presentation_timestamp(&*self.context, self.data.as_ref())
    }
}
