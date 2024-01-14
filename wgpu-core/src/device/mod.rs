use crate::{
    binding_model,
    hal_api::HalApi,
    hub::Hub,
    id::{self},
    identity::{GlobalIdentityHandlerFactory, Input},
    resource::{Buffer, BufferAccessResult},
    resource::{BufferAccessError, BufferMapOperation},
    resource_log, Label, DOWNLEVEL_ERROR_MESSAGE,
};

use arrayvec::ArrayVec;
use hal::Device as _;
use smallvec::SmallVec;
use std::os::raw::c_char;
use thiserror::Error;
use wgt::{BufferAddress, DeviceLostReason, TextureFormat};

use std::{iter, num::NonZeroU32, ptr};

pub mod any_device;
pub(crate) mod bgl;
pub mod global;
mod life;
pub mod queue;
pub mod resource;
#[cfg(any(feature = "trace", feature = "replay"))]
pub mod trace;
pub use {life::WaitIdleError, resource::Device};

pub const SHADER_STAGE_COUNT: usize = hal::MAX_CONCURRENT_SHADER_STAGES;
// Should be large enough for the largest possible texture row. This
// value is enough for a 16k texture with float4 format.
pub(crate) const ZERO_BUFFER_SIZE: BufferAddress = 512 << 10;

const CLEANUP_WAIT_MS: u32 = 5000;

const IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL: &str = "Implicit BindGroupLayout in the Error State";
const ENTRYPOINT_FAILURE_ERROR: &str = "The given EntryPoint is Invalid";

pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum HostMap {
    Read,
    Write,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct AttachmentData<T> {
    pub colors: ArrayVec<Option<T>, { hal::MAX_COLOR_ATTACHMENTS }>,
    pub resolves: ArrayVec<T, { hal::MAX_COLOR_ATTACHMENTS }>,
    pub depth_stencil: Option<T>,
}
impl<T: PartialEq> Eq for AttachmentData<T> {}
impl<T> AttachmentData<T> {
    pub(crate) fn map<U, F: Fn(&T) -> U>(&self, fun: F) -> AttachmentData<U> {
        AttachmentData {
            colors: self.colors.iter().map(|c| c.as_ref().map(&fun)).collect(),
            resolves: self.resolves.iter().map(&fun).collect(),
            depth_stencil: self.depth_stencil.as_ref().map(&fun),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum RenderPassCompatibilityCheckType {
    RenderPipeline,
    RenderBundle,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct RenderPassContext {
    pub attachments: AttachmentData<TextureFormat>,
    pub sample_count: u32,
    pub multiview: Option<NonZeroU32>,
}
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum RenderPassCompatibilityError {
    #[error(
        "Incompatible color attachments at indices {indices:?}: the RenderPass uses textures with formats {expected:?} but the {ty:?} uses attachments with formats {actual:?}",
    )]
    IncompatibleColorAttachment {
        indices: Vec<usize>,
        expected: Vec<Option<TextureFormat>>,
        actual: Vec<Option<TextureFormat>>,
        ty: RenderPassCompatibilityCheckType,
    },
    #[error(
        "Incompatible depth-stencil attachment format: the RenderPass uses a texture with format {expected:?} but the {ty:?} uses an attachment with format {actual:?}",
    )]
    IncompatibleDepthStencilAttachment {
        expected: Option<TextureFormat>,
        actual: Option<TextureFormat>,
        ty: RenderPassCompatibilityCheckType,
    },
    #[error(
        "Incompatible sample count: the RenderPass uses textures with sample count {expected:?} but the {ty:?} uses attachments with format {actual:?}",
    )]
    IncompatibleSampleCount {
        expected: u32,
        actual: u32,
        ty: RenderPassCompatibilityCheckType,
    },
    #[error("Incompatible multiview setting: the RenderPass uses setting {expected:?} but the {ty:?} uses setting {actual:?}")]
    IncompatibleMultiview {
        expected: Option<NonZeroU32>,
        actual: Option<NonZeroU32>,
        ty: RenderPassCompatibilityCheckType,
    },
}

impl RenderPassContext {
    // Assumes the renderpass only contains one subpass
    pub(crate) fn check_compatible(
        &self,
        other: &Self,
        ty: RenderPassCompatibilityCheckType,
    ) -> Result<(), RenderPassCompatibilityError> {
        if self.attachments.colors != other.attachments.colors {
            let indices = self
                .attachments
                .colors
                .iter()
                .zip(&other.attachments.colors)
                .enumerate()
                .filter_map(|(idx, (left, right))| (left != right).then_some(idx))
                .collect();
            return Err(RenderPassCompatibilityError::IncompatibleColorAttachment {
                indices,
                expected: self.attachments.colors.iter().cloned().collect(),
                actual: other.attachments.colors.iter().cloned().collect(),
                ty,
            });
        }
        if self.attachments.depth_stencil != other.attachments.depth_stencil {
            return Err(
                RenderPassCompatibilityError::IncompatibleDepthStencilAttachment {
                    expected: self.attachments.depth_stencil,
                    actual: other.attachments.depth_stencil,
                    ty,
                },
            );
        }
        if self.sample_count != other.sample_count {
            return Err(RenderPassCompatibilityError::IncompatibleSampleCount {
                expected: self.sample_count,
                actual: other.sample_count,
                ty,
            });
        }
        if self.multiview != other.multiview {
            return Err(RenderPassCompatibilityError::IncompatibleMultiview {
                expected: self.multiview,
                actual: other.multiview,
                ty,
            });
        }
        Ok(())
    }
}

pub type BufferMapPendingClosure = (BufferMapOperation, BufferAccessResult);

#[derive(Default)]
pub struct UserClosures {
    pub mappings: Vec<BufferMapPendingClosure>,
    pub submissions: SmallVec<[queue::SubmittedWorkDoneClosure; 1]>,
    pub device_lost_invocations: SmallVec<[DeviceLostInvocation; 1]>,
}

impl UserClosures {
    fn extend(&mut self, other: Self) {
        self.mappings.extend(other.mappings);
        self.submissions.extend(other.submissions);
        self.device_lost_invocations
            .extend(other.device_lost_invocations);
    }

    fn fire(self) {
        // Note: this logic is specifically moved out of `handle_mapping()` in order to
        // have nothing locked by the time we execute users callback code.

        // Mappings _must_ be fired before submissions, as the spec requires all mapping callbacks that are registered before
        // a on_submitted_work_done callback to be fired before the on_submitted_work_done callback.
        for (mut operation, status) in self.mappings {
            if let Some(callback) = operation.callback.take() {
                callback.call(status);
            }
        }
        for closure in self.submissions {
            closure.call();
        }
        for invocation in self.device_lost_invocations {
            invocation
                .closure
                .call(invocation.reason, invocation.message);
        }
    }
}

#[cfg(send_sync)]
pub type DeviceLostCallback = Box<dyn Fn(DeviceLostReason, String) + Send + 'static>;
#[cfg(not(send_sync))]
pub type DeviceLostCallback = Box<dyn Fn(DeviceLostReason, String) + 'static>;

pub struct DeviceLostClosureRust {
    pub callback: DeviceLostCallback,
    consumed: bool,
}

impl Drop for DeviceLostClosureRust {
    fn drop(&mut self) {
        if !self.consumed {
            panic!("DeviceLostClosureRust must be consumed before it is dropped.");
        }
    }
}

#[repr(C)]
pub struct DeviceLostClosureC {
    pub callback: unsafe extern "C" fn(user_data: *mut u8, reason: u8, message: *const c_char),
    pub user_data: *mut u8,
    consumed: bool,
}

#[cfg(send_sync)]
unsafe impl Send for DeviceLostClosureC {}

impl Drop for DeviceLostClosureC {
    fn drop(&mut self) {
        if !self.consumed {
            panic!("DeviceLostClosureC must be consumed before it is dropped.");
        }
    }
}

pub struct DeviceLostClosure {
    // We wrap this so creating the enum in the C variant can be unsafe,
    // allowing our call function to be safe.
    inner: DeviceLostClosureInner,
}

pub struct DeviceLostInvocation {
    closure: DeviceLostClosure,
    reason: DeviceLostReason,
    message: String,
}

enum DeviceLostClosureInner {
    Rust { inner: DeviceLostClosureRust },
    C { inner: DeviceLostClosureC },
}

impl DeviceLostClosure {
    pub fn from_rust(callback: DeviceLostCallback) -> Self {
        let inner = DeviceLostClosureRust {
            callback,
            consumed: false,
        };
        Self {
            inner: DeviceLostClosureInner::Rust { inner },
        }
    }

    /// # Safety
    ///
    /// - The callback pointer must be valid to call with the provided `user_data`
    ///   pointer.
    ///
    /// - Both pointers must point to `'static` data, as the callback may happen at
    ///   an unspecified time.
    pub unsafe fn from_c(mut closure: DeviceLostClosureC) -> Self {
        // Build an inner with the values from closure, ensuring that
        // inner.consumed is false.
        let inner = DeviceLostClosureC {
            callback: closure.callback,
            user_data: closure.user_data,
            consumed: false,
        };

        // Mark the original closure as consumed, so we can safely drop it.
        closure.consumed = true;

        Self {
            inner: DeviceLostClosureInner::C { inner },
        }
    }

    pub(crate) fn call(self, reason: DeviceLostReason, message: String) {
        match self.inner {
            DeviceLostClosureInner::Rust { mut inner } => {
                inner.consumed = true;

                (inner.callback)(reason, message)
            }
            // SAFETY: the contract of the call to from_c says that this unsafe is sound.
            DeviceLostClosureInner::C { mut inner } => unsafe {
                inner.consumed = true;

                // Ensure message is structured as a null-terminated C string. It only
                // needs to live as long as the callback invocation.
                let message = std::ffi::CString::new(message).unwrap();
                (inner.callback)(inner.user_data, reason as u8, message.as_ptr())
            },
        }
    }
}

fn map_buffer<A: HalApi>(
    raw: &A::Device,
    buffer: &Buffer<A>,
    offset: BufferAddress,
    size: BufferAddress,
    kind: HostMap,
) -> Result<ptr::NonNull<u8>, BufferAccessError> {
    let snatch_guard = buffer.device.snatchable_lock.read();
    let raw_buffer = buffer
        .raw(&snatch_guard)
        .ok_or(BufferAccessError::Destroyed)?;
    let mapping = unsafe {
        raw.map_buffer(raw_buffer, offset..offset + size)
            .map_err(DeviceError::from)?
    };

    *buffer.sync_mapped_writes.lock() = match kind {
        HostMap::Read if !mapping.is_coherent => unsafe {
            raw.invalidate_mapped_ranges(raw_buffer, iter::once(offset..offset + size));
            None
        },
        HostMap::Write if !mapping.is_coherent => Some(offset..offset + size),
        _ => None,
    };

    assert_eq!(offset % wgt::COPY_BUFFER_ALIGNMENT, 0);
    assert_eq!(size % wgt::COPY_BUFFER_ALIGNMENT, 0);
    // Zero out uninitialized parts of the mapping. (Spec dictates all resources
    // behave as if they were initialized with zero)
    //
    // If this is a read mapping, ideally we would use a `clear_buffer` command
    // before reading the data from GPU (i.e. `invalidate_range`). However, this
    // would require us to kick off and wait for a command buffer or piggy back
    // on an existing one (the later is likely the only worthwhile option). As
    // reading uninitialized memory isn't a particular important path to
    // support, we instead just initialize the memory here and make sure it is
    // GPU visible, so this happens at max only once for every buffer region.
    //
    // If this is a write mapping zeroing out the memory here is the only
    // reasonable way as all data is pushed to GPU anyways.

    // No need to flush if it is flushed later anyways.
    let zero_init_needs_flush_now =
        mapping.is_coherent && buffer.sync_mapped_writes.lock().is_none();
    let mapped = unsafe { std::slice::from_raw_parts_mut(mapping.ptr.as_ptr(), size as usize) };

    for uninitialized in buffer
        .initialization_status
        .write()
        .drain(offset..(size + offset))
    {
        // The mapping's pointer is already offset, however we track the
        // uninitialized range relative to the buffer's start.
        let fill_range =
            (uninitialized.start - offset) as usize..(uninitialized.end - offset) as usize;
        mapped[fill_range].fill(0);

        if zero_init_needs_flush_now {
            unsafe { raw.flush_mapped_ranges(raw_buffer, iter::once(uninitialized)) };
        }
    }

    Ok(mapping.ptr)
}

pub(crate) struct CommandAllocator<A: HalApi> {
    free_encoders: Vec<A::CommandEncoder>,
}

impl<A: HalApi> CommandAllocator<A> {
    fn acquire_encoder(
        &mut self,
        device: &A::Device,
        queue: &A::Queue,
    ) -> Result<A::CommandEncoder, hal::DeviceError> {
        match self.free_encoders.pop() {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        }
    }

    fn release_encoder(&mut self, encoder: A::CommandEncoder) {
        self.free_encoders.push(encoder);
    }

    fn dispose(self, device: &A::Device) {
        resource_log!(
            "CommandAllocator::dispose encoders {}",
            self.free_encoders.len()
        );
        for cmd_encoder in self.free_encoders {
            unsafe {
                device.destroy_command_encoder(cmd_encoder);
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("Device is invalid")]
pub struct InvalidDevice;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DeviceError {
    #[error("Parent device is invalid.")]
    Invalid,
    #[error("Parent device is lost")]
    Lost,
    #[error("Not enough memory left.")]
    OutOfMemory,
    #[error("Creation of a resource failed for a reason other than running out of memory.")]
    ResourceCreationFailed,
    #[error("QueueId is invalid")]
    InvalidQueueId,
    #[error("Attempt to use a resource with a different device from the one that created it")]
    WrongDevice,
}

impl From<hal::DeviceError> for DeviceError {
    fn from(error: hal::DeviceError) -> Self {
        match error {
            hal::DeviceError::Lost => DeviceError::Lost,
            hal::DeviceError::OutOfMemory => DeviceError::OutOfMemory,
            hal::DeviceError::ResourceCreationFailed => DeviceError::ResourceCreationFailed,
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("Features {0:?} are required but not enabled on the device")]
pub struct MissingFeatures(pub wgt::Features);

#[derive(Clone, Debug, Error)]
#[error(
    "Downlevel flags {0:?} are required but not supported on the device.\n{}",
    DOWNLEVEL_ERROR_MESSAGE
)]
pub struct MissingDownlevelFlags(pub wgt::DownlevelFlags);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImplicitPipelineContext {
    pub root_id: id::PipelineLayoutId,
    pub group_ids: ArrayVec<id::BindGroupLayoutId, { hal::MAX_BIND_GROUPS }>,
}

pub struct ImplicitPipelineIds<'a, G: GlobalIdentityHandlerFactory> {
    pub root_id: Input<G, id::PipelineLayoutId>,
    pub group_ids: &'a [Input<G, id::BindGroupLayoutId>],
}

impl<G: GlobalIdentityHandlerFactory> ImplicitPipelineIds<'_, G> {
    fn prepare<A: HalApi>(self, hub: &Hub<A>) -> ImplicitPipelineContext {
        ImplicitPipelineContext {
            root_id: hub.pipeline_layouts.prepare::<G>(self.root_id).into_id(),
            group_ids: self
                .group_ids
                .iter()
                .map(|id_in| hub.bind_group_layouts.prepare::<G>(*id_in).into_id())
                .collect(),
        }
    }
}
