use crate::{
    binding_model,
    hub::Hub,
    id::{BindGroupLayoutId, PipelineLayoutId},
    resource::{
        Buffer, BufferAccessError, BufferAccessResult, BufferMapOperation, Labeled,
        ResourceErrorIdent,
    },
    snatch::SnatchGuard,
    Label, DOWNLEVEL_ERROR_MESSAGE,
};

use arrayvec::ArrayVec;
use smallvec::SmallVec;
use std::os::raw::c_char;
use thiserror::Error;
use wgt::{BufferAddress, DeviceLostReason, TextureFormat};

use std::num::NonZeroU32;

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

// If a submission is not completed within this time, we go off into UB land.
// See https://github.com/gfx-rs/wgpu/issues/4589. 60s to reduce the chances of this.
const CLEANUP_WAIT_MS: u32 = 60000;

const ENTRYPOINT_FAILURE_ERROR: &str = "The given EntryPoint is Invalid";

pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HostMap {
    Read,
    Write,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct AttachmentData<T> {
    pub colors: ArrayVec<Option<T>, { hal::MAX_COLOR_ATTACHMENTS }>,
    pub resolves: ArrayVec<T, { hal::MAX_COLOR_ATTACHMENTS }>,
    pub depth_stencil: Option<T>,
}
impl<T: PartialEq> Eq for AttachmentData<T> {}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct RenderPassContext {
    pub attachments: AttachmentData<TextureFormat>,
    pub sample_count: u32,
    pub multiview: Option<NonZeroU32>,
}
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum RenderPassCompatibilityError {
    #[error(
        "Incompatible color attachments at indices {indices:?}: the RenderPass uses textures with formats {expected:?} but the {res} uses attachments with formats {actual:?}",
    )]
    IncompatibleColorAttachment {
        indices: Vec<usize>,
        expected: Vec<Option<TextureFormat>>,
        actual: Vec<Option<TextureFormat>>,
        res: ResourceErrorIdent,
    },
    #[error(
        "Incompatible depth-stencil attachment format: the RenderPass uses a texture with format {expected:?} but the {res} uses an attachment with format {actual:?}",
    )]
    IncompatibleDepthStencilAttachment {
        expected: Option<TextureFormat>,
        actual: Option<TextureFormat>,
        res: ResourceErrorIdent,
    },
    #[error(
        "Incompatible sample count: the RenderPass uses textures with sample count {expected:?} but the {res} uses attachments with format {actual:?}",
    )]
    IncompatibleSampleCount {
        expected: u32,
        actual: u32,
        res: ResourceErrorIdent,
    },
    #[error("Incompatible multiview setting: the RenderPass uses setting {expected:?} but the {res} uses setting {actual:?}")]
    IncompatibleMultiview {
        expected: Option<NonZeroU32>,
        actual: Option<NonZeroU32>,
        res: ResourceErrorIdent,
    },
}

impl RenderPassContext {
    // Assumes the renderpass only contains one subpass
    pub(crate) fn check_compatible<T: Labeled>(
        &self,
        other: &Self,
        res: &T,
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
                res: res.error_ident(),
            });
        }
        if self.attachments.depth_stencil != other.attachments.depth_stencil {
            return Err(
                RenderPassCompatibilityError::IncompatibleDepthStencilAttachment {
                    expected: self.attachments.depth_stencil,
                    actual: other.attachments.depth_stencil,
                    res: res.error_ident(),
                },
            );
        }
        if self.sample_count != other.sample_count {
            return Err(RenderPassCompatibilityError::IncompatibleSampleCount {
                expected: self.sample_count,
                actual: other.sample_count,
                res: res.error_ident(),
            });
        }
        if self.multiview != other.multiview {
            return Err(RenderPassCompatibilityError::IncompatibleMultiview {
                expected: self.multiview,
                actual: other.multiview,
                res: res.error_ident(),
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

fn map_buffer(
    raw: &dyn hal::DynDevice,
    buffer: &Buffer,
    offset: BufferAddress,
    size: BufferAddress,
    kind: HostMap,
    snatch_guard: &SnatchGuard,
) -> Result<hal::BufferMapping, BufferAccessError> {
    let raw_buffer = buffer.try_raw(snatch_guard)?;
    let mapping = unsafe {
        raw.map_buffer(raw_buffer, offset..offset + size)
            .map_err(|e| buffer.device.handle_hal_error(e))?
    };

    if !mapping.is_coherent && kind == HostMap::Read {
        #[allow(clippy::single_range_in_vec_init)]
        unsafe {
            raw.invalidate_mapped_ranges(raw_buffer, &[offset..offset + size]);
        }
    }

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

    let mapped = unsafe { std::slice::from_raw_parts_mut(mapping.ptr.as_ptr(), size as usize) };

    // We can't call flush_mapped_ranges in this case, so we can't drain the uninitialized ranges either
    if !mapping.is_coherent
        && kind == HostMap::Read
        && !buffer.usage.contains(wgt::BufferUsages::MAP_WRITE)
    {
        for uninitialized in buffer
            .initialization_status
            .write()
            .uninitialized(offset..(size + offset))
        {
            // The mapping's pointer is already offset, however we track the
            // uninitialized range relative to the buffer's start.
            let fill_range =
                (uninitialized.start - offset) as usize..(uninitialized.end - offset) as usize;
            mapped[fill_range].fill(0);
        }
    } else {
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

            // NOTE: This is only possible when MAPPABLE_PRIMARY_BUFFERS is enabled.
            if !mapping.is_coherent
                && kind == HostMap::Read
                && buffer.usage.contains(wgt::BufferUsages::MAP_WRITE)
            {
                unsafe { raw.flush_mapped_ranges(raw_buffer, &[uninitialized]) };
            }
        }
    }

    Ok(mapping)
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeviceMismatch {
    pub(super) res: ResourceErrorIdent,
    pub(super) res_device: ResourceErrorIdent,
    pub(super) target: Option<ResourceErrorIdent>,
    pub(super) target_device: ResourceErrorIdent,
}

impl std::fmt::Display for DeviceMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{} of {} doesn't match {}",
            self.res_device, self.res, self.target_device
        )?;
        if let Some(target) = self.target.as_ref() {
            write!(f, " of {target}")?;
        }
        Ok(())
    }
}

impl std::error::Error for DeviceMismatch {}

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DeviceError {
    #[error("{0} is invalid.")]
    Invalid(ResourceErrorIdent),
    #[error("Parent device is lost")]
    Lost,
    #[error("Not enough memory left.")]
    OutOfMemory,
    #[error("Creation of a resource failed for a reason other than running out of memory.")]
    ResourceCreationFailed,
    #[error(transparent)]
    DeviceMismatch(#[from] Box<DeviceMismatch>),
}

impl DeviceError {
    /// Only use this function in contexts where there is no `Device`.
    ///
    /// Use [`Device::handle_hal_error`] otherwise.
    pub fn from_hal(error: hal::DeviceError) -> Self {
        match error {
            hal::DeviceError::Lost => Self::Lost,
            hal::DeviceError::OutOfMemory => Self::OutOfMemory,
            hal::DeviceError::ResourceCreationFailed => Self::ResourceCreationFailed,
            hal::DeviceError::Unexpected => Self::Lost,
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImplicitPipelineContext {
    pub root_id: PipelineLayoutId,
    pub group_ids: ArrayVec<BindGroupLayoutId, { hal::MAX_BIND_GROUPS }>,
}

pub struct ImplicitPipelineIds<'a> {
    pub root_id: PipelineLayoutId,
    pub group_ids: &'a [BindGroupLayoutId],
}

impl ImplicitPipelineIds<'_> {
    fn prepare(self, hub: &Hub) -> ImplicitPipelineContext {
        let backend = self.root_id.backend();
        ImplicitPipelineContext {
            root_id: hub
                .pipeline_layouts
                .prepare(backend, Some(self.root_id))
                .id(),
            group_ids: self
                .group_ids
                .iter()
                .map(|id_in| hub.bind_group_layouts.prepare(backend, Some(*id_in)).id())
                .collect(),
        }
    }
}

/// Create a validator with the given validation flags.
pub fn create_validator(
    features: wgt::Features,
    downlevel: wgt::DownlevelFlags,
    flags: naga::valid::ValidationFlags,
) -> naga::valid::Validator {
    use naga::valid::Capabilities as Caps;
    let mut caps = Caps::empty();
    caps.set(
        Caps::PUSH_CONSTANT,
        features.contains(wgt::Features::PUSH_CONSTANTS),
    );
    caps.set(Caps::FLOAT64, features.contains(wgt::Features::SHADER_F64));
    caps.set(
        Caps::PRIMITIVE_INDEX,
        features.contains(wgt::Features::SHADER_PRIMITIVE_INDEX),
    );
    caps.set(
        Caps::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        features
            .contains(wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
    );
    caps.set(
        Caps::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
        features
            .contains(wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING),
    );
    // TODO: This needs a proper wgpu feature
    caps.set(
        Caps::SAMPLER_NON_UNIFORM_INDEXING,
        features
            .contains(wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
    );
    caps.set(
        Caps::STORAGE_TEXTURE_16BIT_NORM_FORMATS,
        features.contains(wgt::Features::TEXTURE_FORMAT_16BIT_NORM),
    );
    caps.set(Caps::MULTIVIEW, features.contains(wgt::Features::MULTIVIEW));
    caps.set(
        Caps::EARLY_DEPTH_TEST,
        features.contains(wgt::Features::SHADER_EARLY_DEPTH_TEST),
    );
    caps.set(
        Caps::SHADER_INT64,
        features.contains(wgt::Features::SHADER_INT64),
    );
    caps.set(
        Caps::SHADER_INT64_ATOMIC_MIN_MAX,
        features.intersects(
            wgt::Features::SHADER_INT64_ATOMIC_MIN_MAX | wgt::Features::SHADER_INT64_ATOMIC_ALL_OPS,
        ),
    );
    caps.set(
        Caps::SHADER_INT64_ATOMIC_ALL_OPS,
        features.contains(wgt::Features::SHADER_INT64_ATOMIC_ALL_OPS),
    );
    caps.set(
        Caps::MULTISAMPLED_SHADING,
        downlevel.contains(wgt::DownlevelFlags::MULTISAMPLED_SHADING),
    );
    caps.set(
        Caps::DUAL_SOURCE_BLENDING,
        features.contains(wgt::Features::DUAL_SOURCE_BLENDING),
    );
    caps.set(
        Caps::CUBE_ARRAY_TEXTURES,
        downlevel.contains(wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES),
    );
    caps.set(
        Caps::SUBGROUP,
        features.intersects(wgt::Features::SUBGROUP | wgt::Features::SUBGROUP_VERTEX),
    );
    caps.set(
        Caps::SUBGROUP_BARRIER,
        features.intersects(wgt::Features::SUBGROUP_BARRIER),
    );
    caps.set(
        Caps::SUBGROUP_VERTEX_STAGE,
        features.contains(wgt::Features::SUBGROUP_VERTEX),
    );

    naga::valid::Validator::new(flags, caps)
}
