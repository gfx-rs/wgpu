use alloc::{
    borrow::Cow,
    boxed::Box,
    string::{String, ToString as _},
    sync::{Arc, Weak},
    vec::Vec,
};
use core::{
    fmt,
    mem::{self, ManuallyDrop},
    num::NonZeroU32,
    sync::atomic::{AtomicBool, Ordering},
};
use hal::ShouldBeNonZeroExt;

use arrayvec::ArrayVec;
use bitflags::Flags;
use smallvec::SmallVec;
use wgt::{
    math::align_to, DeviceLostReason, TextureFormat, TextureSampleType, TextureSelector,
    TextureViewDimension,
};

#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::{self, BindGroup, BindGroupLayout, BindGroupLayoutEntryError},
    command, conv,
    device::{
        bgl, create_validator, life::WaitIdleError, map_buffer, AttachmentData,
        DeviceLostInvocation, HostMap, MissingDownlevelFlags, MissingFeatures, RenderPassContext,
        CLEANUP_WAIT_MS,
    },
    hal_label,
    init_tracker::{
        BufferInitTracker, BufferInitTrackerAction, MemoryInitKind, TextureInitRange,
        TextureInitTrackerAction,
    },
    instance::{Adapter, RequestDeviceError},
    lock::{rank, Mutex, RwLock},
    pipeline,
    pool::ResourcePool,
    resource::{
        self, Buffer, ExternalTexture, Fallible, Labeled, ParentDevice, QuerySet,
        RawResourceAccess, Sampler, StagingBuffer, Texture, TextureView,
        TextureViewNotRenderableReason, Tlas, TrackingData,
    },
    resource_log,
    snatch::{SnatchGuard, SnatchLock, Snatchable},
    timestamp_normalization::TIMESTAMP_NORMALIZATION_BUFFER_USES,
    track::{BindGroupStates, DeviceTracker, TrackerIndexAllocators, UsageScope, UsageScopePool},
    validation::{self, validate_color_attachment_bytes_per_sample},
    weak_vec::WeakVec,
    FastHashMap, LabelHelpers, OnceCellOrLock,
};

use super::{
    queue::Queue, DeviceDescriptor, DeviceError, DeviceLostClosure, UserClosures,
    ENTRYPOINT_FAILURE_ERROR, ZERO_BUFFER_SIZE,
};

#[cfg(supports_64bit_atomics)]
use core::sync::atomic::AtomicU64;
#[cfg(not(supports_64bit_atomics))]
use portable_atomic::AtomicU64;

pub(crate) struct CommandIndices {
    /// The index of the last command submission that was attempted.
    ///
    /// Note that `fence` may never be signalled with this value, if the command
    /// submission failed. If you need to wait for everything running on a
    /// `Queue` to complete, wait for [`last_successful_submission_index`].
    ///
    /// [`last_successful_submission_index`]: Device::last_successful_submission_index
    pub(crate) active_submission_index: hal::FenceValue,
    pub(crate) next_acceleration_structure_build_command_index: u64,
}

/// Parameters provided to shaders via a uniform buffer of the type
/// [`NagaExternalTextureParams`], describing an [`ExternalTexture`] resource
/// binding.
///
/// [`NagaExternalTextureParams`]: naga::SpecialTypes::external_texture_params
/// [`ExternalTexture`]: binding_model::BindingResource::ExternalTexture
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Zeroable, bytemuck::Pod)]
pub struct ExternalTextureParams {
    /// 4x4 column-major matrix with which to convert sampled YCbCr values
    /// to RGBA.
    ///
    /// This is ignored when `num_planes` is 1.
    pub yuv_conversion_matrix: [f32; 16],

    /// 3x3 column-major matrix to transform linear RGB values in the source
    /// color space to linear RGB values in the destination color space. In
    /// combination with [`Self::src_transfer_function`] and
    /// [`Self::dst_transfer_function`] this can be used to ensure that
    /// [`ImageSample`] and [`ImageLoad`] operations return values in the
    /// desired destination color space rather than the source color space of
    /// the underlying planes.
    ///
    /// Includes a padding element after each column.
    ///
    /// [`ImageSample`]: naga::ir::Expression::ImageSample
    /// [`ImageLoad`]: naga::ir::Expression::ImageLoad
    pub gamut_conversion_matrix: [f32; 12],

    /// Transfer function for the source color space. The *inverse* of this
    /// will be applied to decode non-linear RGB to linear RGB in the source
    /// color space.
    pub src_transfer_function: wgt::ExternalTextureTransferFunction,

    /// Transfer function for the destination color space. This will be applied
    /// to encode linear RGB to non-linear RGB in the destination color space.
    pub dst_transfer_function: wgt::ExternalTextureTransferFunction,

    /// Transform to apply to [`ImageSample`] coordinates.
    ///
    /// This is a 3x2 column-major matrix representing an affine transform from
    /// normalized texture coordinates to the normalized coordinates that should
    /// be sampled from the external texture's underlying plane(s).
    ///
    /// This transform may scale, translate, flip, and rotate in 90-degree
    /// increments, but the result of transforming the rectangle (0,0)..(1,1)
    /// must be an axis-aligned rectangle that falls within the bounds of
    /// (0,0)..(1,1).
    ///
    /// [`ImageSample`]: naga::ir::Expression::ImageSample
    pub sample_transform: [f32; 6],

    /// Transform to apply to [`ImageLoad`] coordinates.
    ///
    /// This is a 3x2 column-major matrix representing an affine transform from
    /// non-normalized texel coordinates to the non-normalized coordinates of
    /// the texel that should be loaded from the external texture's underlying
    /// plane 0. For planes 1 and 2, if present, plane 0's coordinates are
    /// scaled according to the textures' relative sizes.
    ///
    /// This transform may scale, translate, flip, and rotate in 90-degree
    /// increments, but the result of transforming the rectangle (0,0)..[`size`]
    /// must be an axis-aligned rectangle that falls within the bounds of
    /// (0,0)..[`size`].
    ///
    /// [`ImageLoad`]: naga::ir::Expression::ImageLoad
    /// [`size`]: Self::size
    pub load_transform: [f32; 6],

    /// Size of the external texture.
    ///
    /// This is the value that should be returned by size queries in shader
    /// code; it does not necessarily match the dimensions of the underlying
    /// texture(s). As a special case, if this is `[0, 0]`, the actual size of
    /// plane 0 should be used instead.
    ///
    /// This must be consistent with [`sample_transform`]: it should be the size
    /// in texels of the rectangle covered by the square (0,0)..(1,1) after
    /// [`sample_transform`] has been applied to it.
    ///
    /// [`sample_transform`]: Self::sample_transform
    pub size: [u32; 2],

    /// Number of planes. 1 indicates a single RGBA plane. 2 indicates a Y
    /// plane and an interleaved CbCr plane. 3 indicates separate Y, Cb, and Cr
    /// planes.
    pub num_planes: u32,
}

impl ExternalTextureParams {
    pub fn from_desc<L>(desc: &wgt::ExternalTextureDescriptor<L>) -> Self {
        let gamut_conversion_matrix = [
            desc.gamut_conversion_matrix[0],
            desc.gamut_conversion_matrix[1],
            desc.gamut_conversion_matrix[2],
            0.0, // padding
            desc.gamut_conversion_matrix[3],
            desc.gamut_conversion_matrix[4],
            desc.gamut_conversion_matrix[5],
            0.0, // padding
            desc.gamut_conversion_matrix[6],
            desc.gamut_conversion_matrix[7],
            desc.gamut_conversion_matrix[8],
            0.0, // padding
        ];

        Self {
            yuv_conversion_matrix: desc.yuv_conversion_matrix,
            gamut_conversion_matrix,
            src_transfer_function: desc.src_transfer_function,
            dst_transfer_function: desc.dst_transfer_function,
            size: [desc.width, desc.height],
            sample_transform: desc.sample_transform,
            load_transform: desc.load_transform,
            num_planes: desc.num_planes() as u32,
        }
    }
}

/// Structure describing a logical device. Some members are internally mutable,
/// stored behind mutexes.
pub struct Device {
    raw: Box<dyn hal::DynDevice>,
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) queue: OnceCellOrLock<Weak<Queue>>,
    pub(crate) zero_buffer: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    /// The `label` from the descriptor used to create the resource.
    label: String,

    pub(crate) command_allocator: command::CommandAllocator,

    pub(crate) command_indices: RwLock<CommandIndices>,

    /// The index of the last successful submission to this device's
    /// [`hal::Queue`].
    ///
    /// Unlike [`active_submission_index`], which is incremented each time
    /// submission is attempted, this is updated only when submission succeeds,
    /// so waiting for this value won't hang waiting for work that was never
    /// submitted.
    ///
    /// [`active_submission_index`]: CommandIndices::active_submission_index
    pub(crate) last_successful_submission_index: hal::AtomicFenceValue,

    // NOTE: if both are needed, the `snatchable_lock` must be consistently acquired before the
    // `fence` lock to avoid deadlocks.
    pub(crate) fence: RwLock<ManuallyDrop<Box<dyn hal::DynFence>>>,
    pub(crate) snatchable_lock: SnatchLock,

    /// Is this device valid? Valid is closely associated with "lose the device",
    /// which can be triggered by various methods, including at the end of device
    /// destroy, and by any GPU errors that cause us to no longer trust the state
    /// of the device. Ideally we would like to fold valid into the storage of
    /// the device itself (for example as an Error enum), but unfortunately we
    /// need to continue to be able to retrieve the device in poll_devices to
    /// determine if it can be dropped. If our internal accesses of devices were
    /// done through ref-counted references and external accesses checked for
    /// Error enums, we wouldn't need this. For now, we need it. All the call
    /// sites where we check it are areas that should be revisited if we start
    /// using ref-counted references for internal access.
    pub(crate) valid: AtomicBool,

    /// Closure to be called on "lose the device". This is invoked directly by
    /// device.lose or by the UserCallbacks returned from maintain when the device
    /// has been destroyed and its queues are empty.
    pub(crate) device_lost_closure: Mutex<Option<DeviceLostClosure>>,

    /// Stores the state of buffers and textures.
    pub(crate) trackers: Mutex<DeviceTracker>,
    pub(crate) tracker_indices: TrackerIndexAllocators,
    /// Pool of bind group layouts, allowing deduplication.
    pub(crate) bgl_pool: ResourcePool<bgl::EntryMap, BindGroupLayout>,
    pub(crate) alignments: hal::Alignments,
    pub(crate) limits: wgt::Limits,
    pub(crate) features: wgt::Features,
    pub(crate) downlevel: wgt::DownlevelCapabilities,
    pub(crate) instance_flags: wgt::InstanceFlags,
    pub(crate) deferred_destroy: Mutex<Vec<DeferredDestroy>>,
    pub(crate) usage_scopes: UsageScopePool,
    pub(crate) indirect_validation: Option<crate::indirect_validation::IndirectValidation>,
    // Optional so that we can late-initialize this after the queue is created.
    pub(crate) timestamp_normalizer:
        OnceCellOrLock<crate::timestamp_normalization::TimestampNormalizer>,
    /// Uniform buffer containing [`ExternalTextureParams`] with values such
    /// that a [`TextureView`] bound to a [`wgt::BindingType::ExternalTexture`]
    /// binding point will be rendered correctly. Intended to be used as the
    /// [`hal::ExternalTextureBinding::params`] field.
    pub(crate) default_external_texture_params_buffer: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    // needs to be dropped last
    #[cfg(feature = "trace")]
    pub(crate) trace: Mutex<Option<trace::Trace>>,
}

pub(crate) enum DeferredDestroy {
    TextureViews(WeakVec<TextureView>),
    BindGroups(WeakVec<BindGroup>),
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("label", &self.label())
            .field("limits", &self.limits)
            .field("features", &self.features)
            .field("downlevel", &self.downlevel)
            .finish()
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        resource_log!("Drop {}", self.error_ident());

        // SAFETY: We are in the Drop impl and we don't use self.zero_buffer anymore after this point.
        let zero_buffer = unsafe { ManuallyDrop::take(&mut self.zero_buffer) };
        // SAFETY: We are in the Drop impl and we don't use
        // self.default_external_texture_params_buffer anymore after this point.
        let default_external_texture_params_buffer =
            unsafe { ManuallyDrop::take(&mut self.default_external_texture_params_buffer) };
        // SAFETY: We are in the Drop impl and we don't use self.fence anymore after this point.
        let fence = unsafe { ManuallyDrop::take(&mut self.fence.write()) };
        if let Some(indirect_validation) = self.indirect_validation.take() {
            indirect_validation.dispose(self.raw.as_ref());
        }
        if let Some(timestamp_normalizer) = self.timestamp_normalizer.take() {
            timestamp_normalizer.dispose(self.raw.as_ref());
        }
        unsafe {
            self.raw.destroy_buffer(zero_buffer);
            self.raw
                .destroy_buffer(default_external_texture_params_buffer);
            self.raw.destroy_fence(fence);
        }
    }
}

impl Device {
    pub(crate) fn raw(&self) -> &dyn hal::DynDevice {
        self.raw.as_ref()
    }
    pub(crate) fn require_features(&self, feature: wgt::Features) -> Result<(), MissingFeatures> {
        if self.features.contains(feature) {
            Ok(())
        } else {
            Err(MissingFeatures(feature))
        }
    }

    pub(crate) fn require_downlevel_flags(
        &self,
        flags: wgt::DownlevelFlags,
    ) -> Result<(), MissingDownlevelFlags> {
        if self.downlevel.flags.contains(flags) {
            Ok(())
        } else {
            Err(MissingDownlevelFlags(flags))
        }
    }
}

impl Device {
    pub(crate) fn new(
        raw_device: Box<dyn hal::DynDevice>,
        adapter: &Arc<Adapter>,
        desc: &DeviceDescriptor,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<Self, DeviceError> {
        #[cfg(not(feature = "trace"))]
        match &desc.trace {
            wgt::Trace::Off => {}
            _ => {
                log::error!("wgpu-core feature 'trace' is not enabled");
            }
        };
        #[cfg(feature = "trace")]
        let trace_dir_name: Option<&std::path::PathBuf> = match &desc.trace {
            wgt::Trace::Off => None,
            wgt::Trace::Directory(d) => Some(d),
            // The enum is non_exhaustive, so we must have a fallback arm (that should be
            // unreachable in practice).
            t => {
                log::error!("unimplemented wgpu_types::Trace variant {t:?}");
                None
            }
        };

        let fence = unsafe { raw_device.create_fence() }.map_err(DeviceError::from_hal)?;

        let command_allocator = command::CommandAllocator::new();

        let rt_uses = if desc
            .required_features
            .intersects(wgt::Features::EXPERIMENTAL_RAY_QUERY)
        {
            wgt::BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT
        } else {
            wgt::BufferUses::empty()
        };

        // Create zeroed buffer used for texture clears (and raytracing if required).
        let zero_buffer = unsafe {
            raw_device.create_buffer(&hal::BufferDescriptor {
                label: hal_label(Some("(wgpu internal) zero init buffer"), instance_flags),
                size: ZERO_BUFFER_SIZE,
                usage: wgt::BufferUses::COPY_SRC | wgt::BufferUses::COPY_DST | rt_uses,
                memory_flags: hal::MemoryFlags::empty(),
            })
        }
        .map_err(DeviceError::from_hal)?;

        let default_external_texture_params_buffer = unsafe {
            raw_device.create_buffer(&hal::BufferDescriptor {
                label: hal_label(
                    Some("(wgpu internal) default external texture params buffer"),
                    instance_flags,
                ),
                size: size_of::<ExternalTextureParams>() as _,
                usage: wgt::BufferUses::COPY_DST | wgt::BufferUses::UNIFORM,
                memory_flags: hal::MemoryFlags::empty(),
            })
        }
        .map_err(DeviceError::from_hal)?;

        let alignments = adapter.raw.capabilities.alignments.clone();
        let downlevel = adapter.raw.capabilities.downlevel.clone();

        let enable_indirect_validation = instance_flags
            .contains(wgt::InstanceFlags::VALIDATION_INDIRECT_CALL)
            && downlevel
                .flags
                .contains(wgt::DownlevelFlags::INDIRECT_EXECUTION);

        let indirect_validation = if enable_indirect_validation {
            Some(crate::indirect_validation::IndirectValidation::new(
                raw_device.as_ref(),
                &desc.required_limits,
                &desc.required_features,
                adapter.backend(),
            )?)
        } else {
            None
        };

        Ok(Self {
            raw: raw_device,
            adapter: adapter.clone(),
            queue: OnceCellOrLock::new(),
            zero_buffer: ManuallyDrop::new(zero_buffer),
            default_external_texture_params_buffer: ManuallyDrop::new(
                default_external_texture_params_buffer,
            ),
            label: desc.label.to_string(),
            command_allocator,
            command_indices: RwLock::new(
                rank::DEVICE_COMMAND_INDICES,
                CommandIndices {
                    active_submission_index: 0,
                    // By starting at one, we can put the result in a NonZeroU64.
                    next_acceleration_structure_build_command_index: 1,
                },
            ),
            last_successful_submission_index: AtomicU64::new(0),
            fence: RwLock::new(rank::DEVICE_FENCE, ManuallyDrop::new(fence)),
            snatchable_lock: unsafe { SnatchLock::new(rank::DEVICE_SNATCHABLE_LOCK) },
            valid: AtomicBool::new(true),
            device_lost_closure: Mutex::new(rank::DEVICE_LOST_CLOSURE, None),
            trackers: Mutex::new(rank::DEVICE_TRACKERS, DeviceTracker::new()),
            tracker_indices: TrackerIndexAllocators::new(),
            bgl_pool: ResourcePool::new(),
            #[cfg(feature = "trace")]
            trace: Mutex::new(
                rank::DEVICE_TRACE,
                trace_dir_name.and_then(|path| match trace::Trace::new(path.clone()) {
                    Ok(mut trace) => {
                        trace.add(trace::Action::Init {
                            desc: wgt::DeviceDescriptor {
                                trace: wgt::Trace::Off,
                                ..desc.clone()
                            },
                            backend: adapter.backend(),
                        });
                        Some(trace)
                    }
                    Err(e) => {
                        log::error!("Unable to start a trace in '{path:?}': {e}");
                        None
                    }
                }),
            ),
            alignments,
            limits: desc.required_limits.clone(),
            features: desc.required_features,
            downlevel,
            instance_flags,
            deferred_destroy: Mutex::new(rank::DEVICE_DEFERRED_DESTROY, Vec::new()),
            usage_scopes: Mutex::new(rank::DEVICE_USAGE_SCOPES, Default::default()),
            timestamp_normalizer: OnceCellOrLock::new(),
            indirect_validation,
        })
    }

    /// Initializes [`Device::default_external_texture_params_buffer`] with
    /// required values such that a [`TextureView`] bound to a
    /// [`wgt::BindingType::ExternalTexture`] binding point will be rendered
    /// correctly.
    fn init_default_external_texture_params_buffer(self: &Arc<Self>) -> Result<(), DeviceError> {
        let data = ExternalTextureParams {
            #[rustfmt::skip]
            yuv_conversion_matrix: [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            #[rustfmt::skip]
            gamut_conversion_matrix: [
                1.0, 0.0, 0.0, /* padding */ 0.0,
                0.0, 1.0, 0.0, /* padding */ 0.0,
                0.0, 0.0, 1.0, /* padding */ 0.0,
            ],
            src_transfer_function: Default::default(),
            dst_transfer_function: Default::default(),
            size: [0, 0],
            #[rustfmt::skip]
            sample_transform: [
                1.0, 0.0,
                0.0, 1.0,
                0.0, 0.0
            ],
            #[rustfmt::skip]
            load_transform: [
                1.0, 0.0,
                0.0, 1.0,
                0.0, 0.0
            ],
            num_planes: 1,
        };
        let mut staging_buffer =
            StagingBuffer::new(self, wgt::BufferSize::new(size_of_val(&data) as _).unwrap())?;
        staging_buffer.write(bytemuck::bytes_of(&data));
        let staging_buffer = staging_buffer.flush();

        let params_buffer = self.default_external_texture_params_buffer.as_ref();
        let queue = self.get_queue().unwrap();
        let mut pending_writes = queue.pending_writes.lock();

        unsafe {
            pending_writes
                .command_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: params_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::MAP_WRITE,
                        to: wgt::BufferUses::COPY_SRC,
                    },
                }]);
            pending_writes.command_encoder.copy_buffer_to_buffer(
                staging_buffer.raw(),
                params_buffer,
                &[hal::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: staging_buffer.size,
                }],
            );
            pending_writes.consume(staging_buffer);
            pending_writes
                .command_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: params_buffer,
                    usage: hal::StateTransition {
                        from: wgt::BufferUses::COPY_DST,
                        to: wgt::BufferUses::UNIFORM,
                    },
                }]);
        }

        Ok(())
    }

    pub fn late_init_resources_with_queue(self: &Arc<Self>) -> Result<(), RequestDeviceError> {
        let queue = self.get_queue().unwrap();

        let timestamp_normalizer = crate::timestamp_normalization::TimestampNormalizer::new(
            self,
            queue.get_timestamp_period(),
        )?;

        self.timestamp_normalizer
            .set(timestamp_normalizer)
            .unwrap_or_else(|_| panic!("Called late_init_resources_with_queue twice"));

        self.init_default_external_texture_params_buffer()?;

        Ok(())
    }

    /// Returns the backend this device is using.
    pub fn backend(&self) -> wgt::Backend {
        self.adapter.backend()
    }

    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }

    pub fn check_is_valid(&self) -> Result<(), DeviceError> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(DeviceError::Lost)
        }
    }

    /// Checks that we are operating within the memory budget reported by the native APIs.
    ///
    /// If we are not, the device gets invalidated.
    ///
    /// The budget might fluctuate over the lifetime of the application, so it should be checked
    /// somewhat frequently.
    pub fn lose_if_oom(&self) {
        let _ = self
            .raw()
            .check_if_oom()
            .map_err(|e| self.handle_hal_error(e));
    }

    pub fn handle_hal_error(&self, error: hal::DeviceError) -> DeviceError {
        match error {
            hal::DeviceError::OutOfMemory
            | hal::DeviceError::Lost
            | hal::DeviceError::Unexpected => {
                self.lose(&error.to_string());
            }
        }
        DeviceError::from_hal(error)
    }

    pub fn handle_hal_error_with_nonfatal_oom(&self, error: hal::DeviceError) -> DeviceError {
        match error {
            hal::DeviceError::OutOfMemory => DeviceError::from_hal(error),
            error => self.handle_hal_error(error),
        }
    }

    /// Run some destroy operations that were deferred.
    ///
    /// Destroying the resources requires taking a write lock on the device's snatch lock,
    /// so a good reason for deferring resource destruction is when we don't know for sure
    /// how risky it is to take the lock (typically, it shouldn't be taken from the drop
    /// implementation of a reference-counted structure).
    /// The snatch lock must not be held while this function is called.
    pub(crate) fn deferred_resource_destruction(&self) {
        let deferred_destroy = mem::take(&mut *self.deferred_destroy.lock());
        for item in deferred_destroy {
            match item {
                DeferredDestroy::TextureViews(views) => {
                    for view in views {
                        let Some(view) = view.upgrade() else {
                            continue;
                        };
                        let Some(raw_view) = view.raw.snatch(&mut self.snatchable_lock.write())
                        else {
                            continue;
                        };

                        resource_log!("Destroy raw {}", view.error_ident());

                        unsafe {
                            self.raw().destroy_texture_view(raw_view);
                        }
                    }
                }
                DeferredDestroy::BindGroups(bind_groups) => {
                    for bind_group in bind_groups {
                        let Some(bind_group) = bind_group.upgrade() else {
                            continue;
                        };
                        let Some(raw_bind_group) =
                            bind_group.raw.snatch(&mut self.snatchable_lock.write())
                        else {
                            continue;
                        };

                        resource_log!("Destroy raw {}", bind_group.error_ident());

                        unsafe {
                            self.raw().destroy_bind_group(raw_bind_group);
                        }
                    }
                }
            }
        }
    }

    pub fn get_queue(&self) -> Option<Arc<Queue>> {
        self.queue.get().as_ref()?.upgrade()
    }

    pub fn set_queue(&self, queue: &Arc<Queue>) {
        assert!(self.queue.set(Arc::downgrade(queue)).is_ok());
    }

    /// Check the current status of the GPU and process any submissions that have
    /// finished.
    ///
    /// The `poll_type` argument tells if this function should wait for a particular
    /// submission index to complete, or if it should just poll the current status.
    ///
    /// This will process _all_ completed submissions, even if the caller only asked
    /// us to poll to a given submission index.
    ///
    /// Return a pair `(closures, result)`, where:
    ///
    /// - `closures` is a list of callbacks that need to be invoked informing the user
    ///   about various things occurring. These happen and should be handled even if
    ///   this function returns an error, hence they are outside of the result.
    ///
    /// - `results` is a boolean indicating the result of the wait operation, including
    ///   if there was a timeout or a validation error.
    pub(crate) fn maintain<'this>(
        &'this self,
        fence: crate::lock::RwLockReadGuard<ManuallyDrop<Box<dyn hal::DynFence>>>,
        poll_type: wgt::PollType<crate::SubmissionIndex>,
        snatch_guard: SnatchGuard,
    ) -> (UserClosures, Result<wgt::PollStatus, WaitIdleError>) {
        profiling::scope!("Device::maintain");

        let mut user_closures = UserClosures::default();

        // If a wait was requested, determine which submission index to wait for.
        let wait_submission_index = match poll_type {
            wgt::PollType::WaitForSubmissionIndex(submission_index) => {
                let last_successful_submission_index = self
                    .last_successful_submission_index
                    .load(Ordering::Acquire);

                if submission_index > last_successful_submission_index {
                    let result = Err(WaitIdleError::WrongSubmissionIndex(
                        submission_index,
                        last_successful_submission_index,
                    ));

                    return (user_closures, result);
                }

                Some(submission_index)
            }
            wgt::PollType::Wait => Some(
                self.last_successful_submission_index
                    .load(Ordering::Acquire),
            ),
            wgt::PollType::Poll => None,
        };

        // Wait for the submission index if requested.
        if let Some(target_submission_index) = wait_submission_index {
            log::trace!("Device::maintain: waiting for submission index {target_submission_index}");

            let wait_result = unsafe {
                self.raw()
                    .wait(fence.as_ref(), target_submission_index, CLEANUP_WAIT_MS)
            };

            // This error match is only about `DeviceErrors`. At this stage we do not care if
            // the wait succeeded or not, and the `Ok(bool)`` variant is ignored.
            if let Err(e) = wait_result {
                let hal_error: WaitIdleError = self.handle_hal_error(e).into();
                return (user_closures, Err(hal_error));
            }
        }

        // Get the currently finished submission index. This may be higher than the requested
        // wait, or it may be less than the requested wait if the wait failed.
        let fence_value_result = unsafe { self.raw().get_fence_value(fence.as_ref()) };
        let current_finished_submission = match fence_value_result {
            Ok(fence_value) => fence_value,
            Err(e) => {
                let hal_error: WaitIdleError = self.handle_hal_error(e).into();
                return (user_closures, Err(hal_error));
            }
        };

        // Maintain all finished submissions on the queue, updating the relevant user closures and collecting if the queue is empty.
        //
        // We don't use the result of the wait here, as we want to progress forward as far as possible
        // and the wait could have been for submissions that finished long ago.
        let mut queue_empty = false;
        if let Some(queue) = self.get_queue() {
            let queue_result = queue.maintain(current_finished_submission, &snatch_guard);
            (
                user_closures.submissions,
                user_closures.mappings,
                user_closures.blas_compact_ready,
                queue_empty,
            ) = queue_result
        };

        // Based on the queue empty status, and the current finished submission index, determine the result of the poll.
        let result = if queue_empty {
            if let Some(wait_submission_index) = wait_submission_index {
                // Assert to ensure that if we received a queue empty status, the fence shows the correct value.
                // This is defensive, as this should never be hit.
                assert!(
                    current_finished_submission >= wait_submission_index,
                    "If the queue is empty, the current submission index ({current_finished_submission}) should be at least the wait submission index ({wait_submission_index})"
                );
            }

            Ok(wgt::PollStatus::QueueEmpty)
        } else if let Some(wait_submission_index) = wait_submission_index {
            // This is theoretically possible to succeed more than checking on the poll result
            // as submissions could have finished in the time between the timeout resolving,
            // the thread getting scheduled again, and us checking the fence value.
            if current_finished_submission >= wait_submission_index {
                Ok(wgt::PollStatus::WaitSucceeded)
            } else {
                Err(WaitIdleError::Timeout)
            }
        } else {
            Ok(wgt::PollStatus::Poll)
        };

        // Detect if we have been destroyed and now need to lose the device.
        //
        // If we are invalid (set at start of destroy) and our queue is empty,
        // and we have a DeviceLostClosure, return the closure to be called by
        // our caller. This will complete the steps for both destroy and for
        // "lose the device".
        let mut should_release_gpu_resource = false;
        if !self.is_valid() && queue_empty {
            // We can release gpu resources associated with this device (but not
            // while holding the life_tracker lock).
            should_release_gpu_resource = true;

            // If we have a DeviceLostClosure, build an invocation with the
            // reason DeviceLostReason::Destroyed and no message.
            if let Some(device_lost_closure) = self.device_lost_closure.lock().take() {
                user_closures
                    .device_lost_invocations
                    .push(DeviceLostInvocation {
                        closure: device_lost_closure,
                        reason: DeviceLostReason::Destroyed,
                        message: String::new(),
                    });
            }
        }

        // Don't hold the locks while calling release_gpu_resources.
        drop(fence);
        drop(snatch_guard);

        if should_release_gpu_resource {
            self.release_gpu_resources();
        }

        (user_closures, result)
    }

    pub(crate) fn create_buffer(
        self: &Arc<Self>,
        desc: &resource::BufferDescriptor,
    ) -> Result<Arc<Buffer>, resource::CreateBufferError> {
        self.check_is_valid()?;

        if desc.size > self.limits.max_buffer_size {
            return Err(resource::CreateBufferError::MaxBufferSize {
                requested: desc.size,
                maximum: self.limits.max_buffer_size,
            });
        }

        if desc
            .usage
            .intersects(wgt::BufferUsages::BLAS_INPUT | wgt::BufferUsages::TLAS_INPUT)
        {
            self.require_features(wgt::Features::EXPERIMENTAL_RAY_QUERY)?;
        }

        if desc.usage.contains(wgt::BufferUsages::INDEX)
            && desc.usage.contains(
                wgt::BufferUsages::VERTEX
                    | wgt::BufferUsages::UNIFORM
                    | wgt::BufferUsages::INDIRECT
                    | wgt::BufferUsages::STORAGE,
            )
        {
            self.require_downlevel_flags(wgt::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER)?;
        }

        if desc.usage.is_empty() || desc.usage.contains_unknown_bits() {
            return Err(resource::CreateBufferError::InvalidUsage(desc.usage));
        }

        if !self
            .features
            .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            use wgt::BufferUsages as Bu;
            let write_mismatch = desc.usage.contains(Bu::MAP_WRITE)
                && !(Bu::MAP_WRITE | Bu::COPY_SRC).contains(desc.usage);
            let read_mismatch = desc.usage.contains(Bu::MAP_READ)
                && !(Bu::MAP_READ | Bu::COPY_DST).contains(desc.usage);
            if write_mismatch || read_mismatch {
                return Err(resource::CreateBufferError::UsageMismatch(desc.usage));
            }
        }

        let mut usage = conv::map_buffer_usage(desc.usage);

        if desc.usage.contains(wgt::BufferUsages::INDIRECT) {
            self.require_downlevel_flags(wgt::DownlevelFlags::INDIRECT_EXECUTION)?;
            // We are going to be reading from it, internally;
            // when validating the content of the buffer
            usage |= wgt::BufferUses::STORAGE_READ_ONLY | wgt::BufferUses::STORAGE_READ_WRITE;
        }

        if desc.usage.contains(wgt::BufferUsages::QUERY_RESOLVE) {
            usage |= TIMESTAMP_NORMALIZATION_BUFFER_USES;
        }

        if desc.mapped_at_creation {
            if desc.size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(resource::CreateBufferError::UnalignedSize);
            }
            if !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                // we are going to be copying into it, internally
                usage |= wgt::BufferUses::COPY_DST;
            }
        } else {
            // We are required to zero out (initialize) all memory. This is done
            // on demand using clear_buffer which requires write transfer usage!
            usage |= wgt::BufferUses::COPY_DST;
        }

        let actual_size = if desc.size == 0 {
            wgt::COPY_BUFFER_ALIGNMENT
        } else if desc.usage.contains(wgt::BufferUsages::VERTEX) {
            // Bumping the size by 1 so that we can bind an empty range at the
            // end of the buffer.
            desc.size + 1
        } else {
            desc.size
        };
        let clear_remainder = actual_size % wgt::COPY_BUFFER_ALIGNMENT;
        let aligned_size = if clear_remainder != 0 {
            actual_size + wgt::COPY_BUFFER_ALIGNMENT - clear_remainder
        } else {
            actual_size
        };

        let hal_desc = hal::BufferDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            size: aligned_size,
            usage,
            memory_flags: hal::MemoryFlags::empty(),
        };
        let buffer = unsafe { self.raw().create_buffer(&hal_desc) }
            .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let timestamp_normalization_bind_group = Snatchable::new(unsafe {
            // SAFETY: The size passed here must not overflow the buffer.
            self.timestamp_normalizer
                .get()
                .unwrap()
                .create_normalization_bind_group(
                    self,
                    &*buffer,
                    desc.label.as_deref(),
                    wgt::BufferSize::new(hal_desc.size).unwrap(),
                    desc.usage,
                )
        }?);

        let indirect_validation_bind_groups =
            self.create_indirect_validation_bind_groups(buffer.as_ref(), desc.size, desc.usage)?;

        let buffer = Buffer {
            raw: Snatchable::new(buffer),
            device: self.clone(),
            usage: desc.usage,
            size: desc.size,
            initialization_status: RwLock::new(
                rank::BUFFER_INITIALIZATION_STATUS,
                BufferInitTracker::new(aligned_size),
            ),
            map_state: Mutex::new(rank::BUFFER_MAP_STATE, resource::BufferMapState::Idle),
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.buffers.clone()),
            bind_groups: Mutex::new(rank::BUFFER_BIND_GROUPS, WeakVec::new()),
            timestamp_normalization_bind_group,
            indirect_validation_bind_groups,
        };

        let buffer = Arc::new(buffer);

        let buffer_use = if !desc.mapped_at_creation {
            wgt::BufferUses::empty()
        } else if desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
            // buffer is mappable, so we are just doing that at start
            let map_size = buffer.size;
            let mapping = if map_size == 0 {
                hal::BufferMapping {
                    ptr: core::ptr::NonNull::dangling(),
                    is_coherent: true,
                }
            } else {
                let snatch_guard: SnatchGuard = self.snatchable_lock.read();
                map_buffer(&buffer, 0, map_size, HostMap::Write, &snatch_guard)?
            };
            *buffer.map_state.lock() = resource::BufferMapState::Active {
                mapping,
                range: 0..map_size,
                host: HostMap::Write,
            };
            wgt::BufferUses::MAP_WRITE
        } else {
            let mut staging_buffer =
                StagingBuffer::new(self, wgt::BufferSize::new(aligned_size).unwrap())?;

            // Zero initialize memory and then mark the buffer as initialized
            // (it's guaranteed that this is the case by the time the buffer is usable)
            staging_buffer.write_zeros();
            buffer.initialization_status.write().drain(0..aligned_size);

            *buffer.map_state.lock() = resource::BufferMapState::Init { staging_buffer };
            wgt::BufferUses::COPY_DST
        };

        self.trackers
            .lock()
            .buffers
            .insert_single(&buffer, buffer_use);

        Ok(buffer)
    }

    pub(crate) fn create_texture_from_hal(
        self: &Arc<Self>,
        hal_texture: Box<dyn hal::DynTexture>,
        desc: &resource::TextureDescriptor,
    ) -> Result<Arc<Texture>, resource::CreateTextureError> {
        let format_features = self
            .describe_format_features(desc.format)
            .map_err(|error| resource::CreateTextureError::MissingFeatures(desc.format, error))?;

        unsafe { self.raw().add_raw_texture(&*hal_texture) };

        let texture = Texture::new(
            self,
            resource::TextureInner::Native { raw: hal_texture },
            conv::map_texture_usage(desc.usage, desc.format.into(), format_features.flags),
            desc,
            format_features,
            resource::TextureClearMode::None,
            false,
        );

        let texture = Arc::new(texture);

        self.trackers
            .lock()
            .textures
            .insert_single(&texture, wgt::TextureUses::UNINITIALIZED);

        Ok(texture)
    }

    /// # Safety
    ///
    /// - `hal_buffer` must have been created on this device.
    /// - `hal_buffer` must have been created respecting `desc` (in particular, the size).
    /// - `hal_buffer` must be initialized.
    /// - `hal_buffer` must not have zero size.
    pub(crate) unsafe fn create_buffer_from_hal(
        self: &Arc<Self>,
        hal_buffer: Box<dyn hal::DynBuffer>,
        desc: &resource::BufferDescriptor,
    ) -> (Fallible<Buffer>, Option<resource::CreateBufferError>) {
        let timestamp_normalization_bind_group = unsafe {
            match self
                .timestamp_normalizer
                .get()
                .unwrap()
                .create_normalization_bind_group(
                    self,
                    &*hal_buffer,
                    desc.label.as_deref(),
                    wgt::BufferSize::new(desc.size).unwrap(),
                    desc.usage,
                ) {
                Ok(bg) => Snatchable::new(bg),
                Err(e) => {
                    return (
                        Fallible::Invalid(Arc::new(desc.label.to_string())),
                        Some(e.into()),
                    )
                }
            }
        };

        let indirect_validation_bind_groups = match self.create_indirect_validation_bind_groups(
            hal_buffer.as_ref(),
            desc.size,
            desc.usage,
        ) {
            Ok(ok) => ok,
            Err(e) => return (Fallible::Invalid(Arc::new(desc.label.to_string())), Some(e)),
        };

        unsafe { self.raw().add_raw_buffer(&*hal_buffer) };

        let buffer = Buffer {
            raw: Snatchable::new(hal_buffer),
            device: self.clone(),
            usage: desc.usage,
            size: desc.size,
            initialization_status: RwLock::new(
                rank::BUFFER_INITIALIZATION_STATUS,
                BufferInitTracker::new(0),
            ),
            map_state: Mutex::new(rank::BUFFER_MAP_STATE, resource::BufferMapState::Idle),
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.buffers.clone()),
            bind_groups: Mutex::new(rank::BUFFER_BIND_GROUPS, WeakVec::new()),
            timestamp_normalization_bind_group,
            indirect_validation_bind_groups,
        };

        let buffer = Arc::new(buffer);

        self.trackers
            .lock()
            .buffers
            .insert_single(&buffer, wgt::BufferUses::empty());

        (Fallible::Valid(buffer), None)
    }

    fn create_indirect_validation_bind_groups(
        &self,
        raw_buffer: &dyn hal::DynBuffer,
        buffer_size: u64,
        usage: wgt::BufferUsages,
    ) -> Result<Snatchable<crate::indirect_validation::BindGroups>, resource::CreateBufferError>
    {
        if !usage.contains(wgt::BufferUsages::INDIRECT) {
            return Ok(Snatchable::empty());
        }

        let Some(ref indirect_validation) = self.indirect_validation else {
            return Ok(Snatchable::empty());
        };

        let bind_groups = crate::indirect_validation::BindGroups::new(
            indirect_validation,
            self,
            buffer_size,
            raw_buffer,
        )
        .map_err(resource::CreateBufferError::IndirectValidationBindGroup)?;

        if let Some(bind_groups) = bind_groups {
            Ok(Snatchable::new(bind_groups))
        } else {
            Ok(Snatchable::empty())
        }
    }

    pub(crate) fn create_texture(
        self: &Arc<Self>,
        desc: &resource::TextureDescriptor,
    ) -> Result<Arc<Texture>, resource::CreateTextureError> {
        use resource::{CreateTextureError, TextureDimensionError};

        self.check_is_valid()?;

        if desc.usage.is_empty() || desc.usage.contains_unknown_bits() {
            return Err(CreateTextureError::InvalidUsage(desc.usage));
        }

        conv::check_texture_dimension_size(
            desc.dimension,
            desc.size,
            desc.sample_count,
            &self.limits,
        )?;

        if desc.dimension != wgt::TextureDimension::D2 {
            // Depth textures can only be 2D
            if desc.format.is_depth_stencil_format() {
                return Err(CreateTextureError::InvalidDepthDimension(
                    desc.dimension,
                    desc.format,
                ));
            }
        }

        if desc.dimension != wgt::TextureDimension::D2
            && desc.dimension != wgt::TextureDimension::D3
        {
            // Compressed textures can only be 2D or 3D
            if desc.format.is_compressed() {
                return Err(CreateTextureError::InvalidCompressedDimension(
                    desc.dimension,
                    desc.format,
                ));
            }

            // Renderable textures can only be 2D or 3D
            if desc.usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) {
                return Err(CreateTextureError::InvalidDimensionUsages(
                    wgt::TextureUsages::RENDER_ATTACHMENT,
                    desc.dimension,
                ));
            }
        }

        if desc.format.is_compressed() {
            let (block_width, block_height) = desc.format.block_dimensions();

            if desc.size.width % block_width != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::NotMultipleOfBlockWidth {
                        width: desc.size.width,
                        block_width,
                        format: desc.format,
                    },
                ));
            }

            if desc.size.height % block_height != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::NotMultipleOfBlockHeight {
                        height: desc.size.height,
                        block_height,
                        format: desc.format,
                    },
                ));
            }

            if desc.dimension == wgt::TextureDimension::D3 {
                // Only BCn formats with Sliced 3D feature can be used for 3D textures
                if desc.format.is_bcn() {
                    self.require_features(wgt::Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
                        .map_err(|error| CreateTextureError::MissingFeatures(desc.format, error))?;
                } else if desc.format.is_astc() {
                    self.require_features(wgt::Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
                        .map_err(|error| CreateTextureError::MissingFeatures(desc.format, error))?;
                } else {
                    return Err(CreateTextureError::InvalidCompressedDimension(
                        desc.dimension,
                        desc.format,
                    ));
                }
            }
        }

        {
            let (width_multiple, height_multiple) = desc.format.size_multiple_requirement();

            if desc.size.width % width_multiple != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::WidthNotMultipleOf {
                        width: desc.size.width,
                        multiple: width_multiple,
                        format: desc.format,
                    },
                ));
            }

            if desc.size.height % height_multiple != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::HeightNotMultipleOf {
                        height: desc.size.height,
                        multiple: height_multiple,
                        format: desc.format,
                    },
                ));
            }
        }

        let format_features = self
            .describe_format_features(desc.format)
            .map_err(|error| CreateTextureError::MissingFeatures(desc.format, error))?;

        if desc.sample_count > 1 {
            if desc.mip_level_count != 1 {
                return Err(CreateTextureError::InvalidMipLevelCount {
                    requested: desc.mip_level_count,
                    maximum: 1,
                });
            }

            if desc.size.depth_or_array_layers != 1 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::MultisampledDepthOrArrayLayer(
                        desc.size.depth_or_array_layers,
                    ),
                ));
            }

            if desc.usage.contains(wgt::TextureUsages::STORAGE_BINDING) {
                return Err(CreateTextureError::InvalidMultisampledStorageBinding);
            }

            if !desc.usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) {
                return Err(CreateTextureError::MultisampledNotRenderAttachment);
            }

            if !format_features.flags.intersects(
                wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4
                    | wgt::TextureFormatFeatureFlags::MULTISAMPLE_X2
                    | wgt::TextureFormatFeatureFlags::MULTISAMPLE_X8
                    | wgt::TextureFormatFeatureFlags::MULTISAMPLE_X16,
            ) {
                return Err(CreateTextureError::InvalidMultisampledFormat(desc.format));
            }

            if !format_features
                .flags
                .sample_count_supported(desc.sample_count)
            {
                return Err(CreateTextureError::InvalidSampleCount(
                    desc.sample_count,
                    desc.format,
                    desc.format
                        .guaranteed_format_features(self.features)
                        .flags
                        .supported_sample_counts(),
                    self.adapter
                        .get_texture_format_features(desc.format)
                        .flags
                        .supported_sample_counts(),
                ));
            };
        }

        let mips = desc.mip_level_count;
        let max_levels_allowed = desc.size.max_mips(desc.dimension).min(hal::MAX_MIP_LEVELS);
        if mips == 0 || mips > max_levels_allowed {
            return Err(CreateTextureError::InvalidMipLevelCount {
                requested: mips,
                maximum: max_levels_allowed,
            });
        }

        let missing_allowed_usages = desc.usage - format_features.allowed_usages;
        if !missing_allowed_usages.is_empty() {
            // detect downlevel incompatibilities
            let wgpu_allowed_usages = desc
                .format
                .guaranteed_format_features(self.features)
                .allowed_usages;
            let wgpu_missing_usages = desc.usage - wgpu_allowed_usages;
            return Err(CreateTextureError::InvalidFormatUsages(
                missing_allowed_usages,
                desc.format,
                wgpu_missing_usages.is_empty(),
            ));
        }

        let mut hal_view_formats = Vec::new();
        for format in desc.view_formats.iter() {
            if desc.format == *format {
                continue;
            }
            if desc.format.remove_srgb_suffix() != format.remove_srgb_suffix() {
                return Err(CreateTextureError::InvalidViewFormat(*format, desc.format));
            }
            hal_view_formats.push(*format);
        }
        if !hal_view_formats.is_empty() {
            self.require_downlevel_flags(wgt::DownlevelFlags::VIEW_FORMATS)?;
        }

        let hal_usage = conv::map_texture_usage_for_texture(desc, &format_features);

        let hal_desc = hal::TextureDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: hal_usage,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: hal_view_formats,
        };

        let raw_texture = unsafe { self.raw().create_texture(&hal_desc) }
            .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let clear_mode = if hal_usage
            .intersects(wgt::TextureUses::DEPTH_STENCIL_WRITE | wgt::TextureUses::COLOR_TARGET)
            && desc.dimension == wgt::TextureDimension::D2
        {
            let (is_color, usage) = if desc.format.is_depth_stencil_format() {
                (false, wgt::TextureUses::DEPTH_STENCIL_WRITE)
            } else {
                (true, wgt::TextureUses::COLOR_TARGET)
            };

            let clear_label = hal_label(
                Some("(wgpu internal) clear texture view"),
                self.instance_flags,
            );

            let mut clear_views = SmallVec::new();
            for mip_level in 0..desc.mip_level_count {
                for array_layer in 0..desc.size.depth_or_array_layers {
                    macro_rules! push_clear_view {
                        ($format:expr, $aspect:expr) => {
                            let desc = hal::TextureViewDescriptor {
                                label: clear_label,
                                format: $format,
                                dimension: TextureViewDimension::D2,
                                usage,
                                range: wgt::ImageSubresourceRange {
                                    aspect: $aspect,
                                    base_mip_level: mip_level,
                                    mip_level_count: Some(1),
                                    base_array_layer: array_layer,
                                    array_layer_count: Some(1),
                                },
                            };
                            clear_views.push(ManuallyDrop::new(
                                unsafe {
                                    self.raw().create_texture_view(raw_texture.as_ref(), &desc)
                                }
                                .map_err(|e| self.handle_hal_error(e))?,
                            ));
                        };
                    }

                    if let Some(planes) = desc.format.planes() {
                        for plane in 0..planes {
                            let aspect = wgt::TextureAspect::from_plane(plane).unwrap();
                            let format = desc.format.aspect_specific_format(aspect).unwrap();
                            push_clear_view!(format, aspect);
                        }
                    } else {
                        push_clear_view!(desc.format, wgt::TextureAspect::All);
                    }
                }
            }
            resource::TextureClearMode::RenderPass {
                clear_views,
                is_color,
            }
        } else {
            resource::TextureClearMode::BufferCopy
        };

        let texture = Texture::new(
            self,
            resource::TextureInner::Native { raw: raw_texture },
            hal_usage,
            desc,
            format_features,
            clear_mode,
            true,
        );

        let texture = Arc::new(texture);

        self.trackers
            .lock()
            .textures
            .insert_single(&texture, wgt::TextureUses::UNINITIALIZED);

        Ok(texture)
    }

    pub(crate) fn create_texture_view(
        self: &Arc<Self>,
        texture: &Arc<Texture>,
        desc: &resource::TextureViewDescriptor,
    ) -> Result<Arc<TextureView>, resource::CreateTextureViewError> {
        self.check_is_valid()?;

        let snatch_guard = texture.device.snatchable_lock.read();

        let texture_raw = texture.try_raw(&snatch_guard)?;

        // resolve TextureViewDescriptor defaults
        // https://gpuweb.github.io/gpuweb/#abstract-opdef-resolving-gputextureviewdescriptor-defaults
        let resolved_format = desc.format.unwrap_or_else(|| {
            texture
                .desc
                .format
                .aspect_specific_format(desc.range.aspect)
                .unwrap_or(texture.desc.format)
        });

        let resolved_dimension = desc
            .dimension
            .unwrap_or_else(|| match texture.desc.dimension {
                wgt::TextureDimension::D1 => TextureViewDimension::D1,
                wgt::TextureDimension::D2 => {
                    if texture.desc.array_layer_count() == 1 {
                        TextureViewDimension::D2
                    } else {
                        TextureViewDimension::D2Array
                    }
                }
                wgt::TextureDimension::D3 => TextureViewDimension::D3,
            });

        let resolved_mip_level_count = desc.range.mip_level_count.unwrap_or_else(|| {
            texture
                .desc
                .mip_level_count
                .saturating_sub(desc.range.base_mip_level)
        });

        let resolved_array_layer_count =
            desc.range
                .array_layer_count
                .unwrap_or_else(|| match resolved_dimension {
                    TextureViewDimension::D1
                    | TextureViewDimension::D2
                    | TextureViewDimension::D3 => 1,
                    TextureViewDimension::Cube => 6,
                    TextureViewDimension::D2Array | TextureViewDimension::CubeArray => texture
                        .desc
                        .array_layer_count()
                        .saturating_sub(desc.range.base_array_layer),
                });

        let resolved_usage = {
            let usage = desc.usage.unwrap_or(wgt::TextureUsages::empty());
            if usage.is_empty() {
                texture.desc.usage
            } else if texture.desc.usage.contains(usage) {
                usage
            } else {
                return Err(resource::CreateTextureViewError::InvalidTextureViewUsage {
                    view: usage,
                    texture: texture.desc.usage,
                });
            }
        };

        let format_features = self.describe_format_features(resolved_format)?;
        let allowed_format_usages = format_features.allowed_usages;
        if resolved_usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT)
            && !allowed_format_usages.contains(wgt::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(
                resource::CreateTextureViewError::TextureViewFormatNotRenderable(resolved_format),
            );
        }

        if resolved_usage.contains(wgt::TextureUsages::STORAGE_BINDING)
            && !allowed_format_usages.contains(wgt::TextureUsages::STORAGE_BINDING)
        {
            return Err(
                resource::CreateTextureViewError::TextureViewFormatNotStorage(resolved_format),
            );
        }

        // validate TextureViewDescriptor

        let aspects = hal::FormatAspects::new(texture.desc.format, desc.range.aspect);
        if aspects.is_empty() {
            return Err(resource::CreateTextureViewError::InvalidAspect {
                texture_format: texture.desc.format,
                requested_aspect: desc.range.aspect,
            });
        }

        let format_is_good = if desc.range.aspect == wgt::TextureAspect::All {
            resolved_format == texture.desc.format
                || texture.desc.view_formats.contains(&resolved_format)
        } else {
            Some(resolved_format)
                == texture
                    .desc
                    .format
                    .aspect_specific_format(desc.range.aspect)
        };
        if !format_is_good {
            return Err(resource::CreateTextureViewError::FormatReinterpretation {
                texture: texture.desc.format,
                view: resolved_format,
            });
        }

        // check if multisampled texture is seen as anything but 2D
        if texture.desc.sample_count > 1 && resolved_dimension != TextureViewDimension::D2 {
            return Err(
                resource::CreateTextureViewError::InvalidMultisampledTextureViewDimension(
                    resolved_dimension,
                ),
            );
        }

        // check if the dimension is compatible with the texture
        if texture.desc.dimension != resolved_dimension.compatible_texture_dimension() {
            return Err(
                resource::CreateTextureViewError::InvalidTextureViewDimension {
                    view: resolved_dimension,
                    texture: texture.desc.dimension,
                },
            );
        }

        match resolved_dimension {
            TextureViewDimension::D1 | TextureViewDimension::D2 | TextureViewDimension::D3 => {
                if resolved_array_layer_count != 1 {
                    return Err(resource::CreateTextureViewError::InvalidArrayLayerCount {
                        requested: resolved_array_layer_count,
                        dim: resolved_dimension,
                    });
                }
            }
            TextureViewDimension::Cube => {
                if resolved_array_layer_count != 6 {
                    return Err(
                        resource::CreateTextureViewError::InvalidCubemapTextureDepth {
                            depth: resolved_array_layer_count,
                        },
                    );
                }
            }
            TextureViewDimension::CubeArray => {
                if resolved_array_layer_count % 6 != 0 {
                    return Err(
                        resource::CreateTextureViewError::InvalidCubemapArrayTextureDepth {
                            depth: resolved_array_layer_count,
                        },
                    );
                }
            }
            _ => {}
        }

        match resolved_dimension {
            TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                if texture.desc.size.width != texture.desc.size.height {
                    return Err(resource::CreateTextureViewError::InvalidCubeTextureViewSize);
                }
            }
            _ => {}
        }

        if resolved_mip_level_count == 0 {
            return Err(resource::CreateTextureViewError::ZeroMipLevelCount);
        }

        let mip_level_end = desc
            .range
            .base_mip_level
            .saturating_add(resolved_mip_level_count);

        let level_end = texture.desc.mip_level_count;
        if mip_level_end > level_end {
            return Err(resource::CreateTextureViewError::TooManyMipLevels {
                requested: mip_level_end,
                total: level_end,
            });
        }

        if resolved_array_layer_count == 0 {
            return Err(resource::CreateTextureViewError::ZeroArrayLayerCount);
        }

        let array_layer_end = desc
            .range
            .base_array_layer
            .saturating_add(resolved_array_layer_count);

        let layer_end = texture.desc.array_layer_count();
        if array_layer_end > layer_end {
            return Err(resource::CreateTextureViewError::TooManyArrayLayers {
                requested: array_layer_end,
                total: layer_end,
            });
        };

        // https://gpuweb.github.io/gpuweb/#abstract-opdef-renderable-texture-view
        let render_extent = 'error: {
            if !resolved_usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) {
                break 'error Err(TextureViewNotRenderableReason::Usage(resolved_usage));
            }

            let allowed_view_dimensions = [
                TextureViewDimension::D2,
                TextureViewDimension::D2Array,
                TextureViewDimension::D3,
            ];
            if !allowed_view_dimensions.contains(&resolved_dimension) {
                break 'error Err(TextureViewNotRenderableReason::Dimension(
                    resolved_dimension,
                ));
            }

            if resolved_mip_level_count != 1 {
                break 'error Err(TextureViewNotRenderableReason::MipLevelCount(
                    resolved_mip_level_count,
                ));
            }

            if resolved_array_layer_count != 1
                && !(self.features.contains(wgt::Features::MULTIVIEW))
            {
                break 'error Err(TextureViewNotRenderableReason::ArrayLayerCount(
                    resolved_array_layer_count,
                ));
            }

            if aspects != hal::FormatAspects::from(texture.desc.format) {
                break 'error Err(TextureViewNotRenderableReason::Aspects(aspects));
            }

            Ok(texture
                .desc
                .compute_render_extent(desc.range.base_mip_level))
        };

        // filter the usages based on the other criteria
        let usage = {
            let resolved_hal_usage = conv::map_texture_usage(
                resolved_usage,
                resolved_format.into(),
                format_features.flags,
            );
            let mask_copy = !(wgt::TextureUses::COPY_SRC | wgt::TextureUses::COPY_DST);
            let mask_dimension = match resolved_dimension {
                TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                    wgt::TextureUses::RESOURCE
                }
                TextureViewDimension::D3 => {
                    wgt::TextureUses::RESOURCE
                        | wgt::TextureUses::STORAGE_READ_ONLY
                        | wgt::TextureUses::STORAGE_WRITE_ONLY
                        | wgt::TextureUses::STORAGE_READ_WRITE
                }
                _ => wgt::TextureUses::all(),
            };
            let mask_mip_level = if resolved_mip_level_count == 1 {
                wgt::TextureUses::all()
            } else {
                wgt::TextureUses::RESOURCE
            };
            resolved_hal_usage & mask_copy & mask_dimension & mask_mip_level
        };

        // use the combined depth-stencil format for the view
        let format = if resolved_format.is_depth_stencil_component(texture.desc.format) {
            texture.desc.format
        } else {
            resolved_format
        };

        let resolved_range = wgt::ImageSubresourceRange {
            aspect: desc.range.aspect,
            base_mip_level: desc.range.base_mip_level,
            mip_level_count: Some(resolved_mip_level_count),
            base_array_layer: desc.range.base_array_layer,
            array_layer_count: Some(resolved_array_layer_count),
        };

        let hal_desc = hal::TextureViewDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            format,
            dimension: resolved_dimension,
            usage,
            range: resolved_range,
        };

        let raw = unsafe { self.raw().create_texture_view(texture_raw, &hal_desc) }
            .map_err(|e| self.handle_hal_error(e))?;

        let selector = TextureSelector {
            mips: desc.range.base_mip_level..mip_level_end,
            layers: desc.range.base_array_layer..array_layer_end,
        };

        let view = TextureView {
            raw: Snatchable::new(raw),
            parent: texture.clone(),
            device: self.clone(),
            desc: resource::HalTextureViewDescriptor {
                texture_format: texture.desc.format,
                format: resolved_format,
                dimension: resolved_dimension,
                usage: resolved_usage,
                range: resolved_range,
            },
            format_features: texture.format_features,
            render_extent,
            samples: texture.desc.sample_count,
            selector,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.texture_views.clone()),
        };

        let view = Arc::new(view);

        {
            let mut views = texture.views.lock();
            views.push(Arc::downgrade(&view));
        }

        Ok(view)
    }

    pub(crate) fn create_external_texture(
        self: &Arc<Self>,
        desc: &resource::ExternalTextureDescriptor,
        planes: &[Arc<TextureView>],
    ) -> Result<Arc<ExternalTexture>, resource::CreateExternalTextureError> {
        use resource::CreateExternalTextureError;
        self.require_features(wgt::Features::EXTERNAL_TEXTURE)?;
        self.check_is_valid()?;

        if desc.num_planes() != planes.len() {
            return Err(CreateExternalTextureError::IncorrectPlaneCount {
                format: desc.format,
                expected: desc.num_planes(),
                provided: planes.len(),
            });
        }

        let planes = planes
            .iter()
            .enumerate()
            .map(|(i, plane)| {
                if plane.samples != 1 {
                    return Err(CreateExternalTextureError::InvalidPlaneMultisample(
                        plane.samples,
                    ));
                }

                let sample_type = plane
                    .desc
                    .format
                    .sample_type(Some(plane.desc.range.aspect), Some(self.features))
                    .unwrap();
                if !matches!(sample_type, TextureSampleType::Float { filterable: true }) {
                    return Err(CreateExternalTextureError::InvalidPlaneSampleType {
                        format: plane.desc.format,
                        sample_type,
                    });
                }

                if plane.desc.dimension != TextureViewDimension::D2 {
                    return Err(CreateExternalTextureError::InvalidPlaneDimension(
                        plane.desc.dimension,
                    ));
                }

                let expected_components = match desc.format {
                    wgt::ExternalTextureFormat::Rgba => 4,
                    wgt::ExternalTextureFormat::Nv12 => match i {
                        0 => 1,
                        1 => 2,
                        _ => unreachable!(),
                    },
                    wgt::ExternalTextureFormat::Yu12 => 1,
                };
                if plane.desc.format.components() != expected_components {
                    return Err(CreateExternalTextureError::InvalidPlaneFormat {
                        format: desc.format,
                        plane: i,
                        expected: expected_components,
                        provided: plane.desc.format,
                    });
                }

                plane.check_usage(wgt::TextureUsages::TEXTURE_BINDING)?;
                Ok(plane.clone())
            })
            .collect::<Result<_, _>>()?;

        let params_data = ExternalTextureParams::from_desc(desc);
        let label = desc.label.as_ref().map(|l| alloc::format!("{l} params"));
        let params_desc = resource::BufferDescriptor {
            label: label.map(Cow::Owned),
            size: size_of_val(&params_data) as wgt::BufferAddress,
            usage: wgt::BufferUsages::UNIFORM | wgt::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let params = self.create_buffer(&params_desc)?;
        self.get_queue().unwrap().write_buffer(
            Fallible::Valid(params.clone()),
            0,
            bytemuck::bytes_of(&params_data),
        )?;

        let external_texture = ExternalTexture {
            device: self.clone(),
            planes,
            params,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.external_textures.clone()),
        };
        let external_texture = Arc::new(external_texture);

        Ok(external_texture)
    }

    pub(crate) fn create_sampler(
        self: &Arc<Self>,
        desc: &resource::SamplerDescriptor,
    ) -> Result<Arc<Sampler>, resource::CreateSamplerError> {
        self.check_is_valid()?;

        if desc
            .address_modes
            .iter()
            .any(|am| am == &wgt::AddressMode::ClampToBorder)
        {
            self.require_features(wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER)?;
        }

        if desc.border_color == Some(wgt::SamplerBorderColor::Zero) {
            self.require_features(wgt::Features::ADDRESS_MODE_CLAMP_TO_ZERO)?;
        }

        if desc.lod_min_clamp < 0.0 {
            return Err(resource::CreateSamplerError::InvalidLodMinClamp(
                desc.lod_min_clamp,
            ));
        }
        if desc.lod_max_clamp < desc.lod_min_clamp {
            return Err(resource::CreateSamplerError::InvalidLodMaxClamp {
                lod_min_clamp: desc.lod_min_clamp,
                lod_max_clamp: desc.lod_max_clamp,
            });
        }

        if desc.anisotropy_clamp < 1 {
            return Err(resource::CreateSamplerError::InvalidAnisotropy(
                desc.anisotropy_clamp,
            ));
        }

        if desc.anisotropy_clamp != 1 {
            if !matches!(desc.min_filter, wgt::FilterMode::Linear) {
                return Err(
                    resource::CreateSamplerError::InvalidFilterModeWithAnisotropy {
                        filter_type: resource::SamplerFilterErrorType::MinFilter,
                        filter_mode: desc.min_filter,
                        anisotropic_clamp: desc.anisotropy_clamp,
                    },
                );
            }
            if !matches!(desc.mag_filter, wgt::FilterMode::Linear) {
                return Err(
                    resource::CreateSamplerError::InvalidFilterModeWithAnisotropy {
                        filter_type: resource::SamplerFilterErrorType::MagFilter,
                        filter_mode: desc.mag_filter,
                        anisotropic_clamp: desc.anisotropy_clamp,
                    },
                );
            }
            if !matches!(desc.mipmap_filter, wgt::FilterMode::Linear) {
                return Err(
                    resource::CreateSamplerError::InvalidFilterModeWithAnisotropy {
                        filter_type: resource::SamplerFilterErrorType::MipmapFilter,
                        filter_mode: desc.mipmap_filter,
                        anisotropic_clamp: desc.anisotropy_clamp,
                    },
                );
            }
        }

        let anisotropy_clamp = if self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::ANISOTROPIC_FILTERING)
        {
            // Clamp anisotropy clamp to [1, 16] per the wgpu-hal interface
            desc.anisotropy_clamp.min(16)
        } else {
            // If it isn't supported, set this unconditionally to 1
            1
        };

        //TODO: check for wgt::DownlevelFlags::COMPARISON_SAMPLERS

        let hal_desc = hal::SamplerDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            address_modes: desc.address_modes,
            mag_filter: desc.mag_filter,
            min_filter: desc.min_filter,
            mipmap_filter: desc.mipmap_filter,
            lod_clamp: desc.lod_min_clamp..desc.lod_max_clamp,
            compare: desc.compare,
            anisotropy_clamp,
            border_color: desc.border_color,
        };

        let raw = unsafe { self.raw().create_sampler(&hal_desc) }
            .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let sampler = Sampler {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.samplers.clone()),
            comparison: desc.compare.is_some(),
            filtering: desc.min_filter == wgt::FilterMode::Linear
                || desc.mag_filter == wgt::FilterMode::Linear
                || desc.mipmap_filter == wgt::FilterMode::Linear,
        };

        let sampler = Arc::new(sampler);

        Ok(sampler)
    }

    pub(crate) fn create_shader_module<'a>(
        self: &Arc<Self>,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: pipeline::ShaderModuleSource<'a>,
    ) -> Result<Arc<pipeline::ShaderModule>, pipeline::CreateShaderModuleError> {
        self.check_is_valid()?;

        let (module, source) = match source {
            #[cfg(feature = "wgsl")]
            pipeline::ShaderModuleSource::Wgsl(code) => {
                profiling::scope!("naga::front::wgsl::parse_str");
                let module = naga::front::wgsl::parse_str(&code).map_err(|inner| {
                    pipeline::CreateShaderModuleError::Parsing(naga::error::ShaderError {
                        source: code.to_string(),
                        label: desc.label.as_ref().map(|l| l.to_string()),
                        inner: Box::new(inner),
                    })
                })?;
                (Cow::Owned(module), code.into_owned())
            }
            #[cfg(feature = "spirv")]
            pipeline::ShaderModuleSource::SpirV(spv, options) => {
                let parser = naga::front::spv::Frontend::new(spv.iter().cloned(), &options);
                profiling::scope!("naga::front::spv::Frontend");
                let module = parser.parse().map_err(|inner| {
                    pipeline::CreateShaderModuleError::ParsingSpirV(naga::error::ShaderError {
                        source: String::new(),
                        label: desc.label.as_ref().map(|l| l.to_string()),
                        inner: Box::new(inner),
                    })
                })?;
                (Cow::Owned(module), String::new())
            }
            #[cfg(feature = "glsl")]
            pipeline::ShaderModuleSource::Glsl(code, options) => {
                let mut parser = naga::front::glsl::Frontend::default();
                profiling::scope!("naga::front::glsl::Frontend.parse");
                let module = parser.parse(&options, &code).map_err(|inner| {
                    pipeline::CreateShaderModuleError::ParsingGlsl(naga::error::ShaderError {
                        source: code.to_string(),
                        label: desc.label.as_ref().map(|l| l.to_string()),
                        inner: Box::new(inner),
                    })
                })?;
                (Cow::Owned(module), code.into_owned())
            }
            pipeline::ShaderModuleSource::Naga(module) => (module, String::new()),
            pipeline::ShaderModuleSource::Dummy(_) => panic!("found `ShaderModuleSource::Dummy`"),
        };
        for (_, var) in module.global_variables.iter() {
            match var.binding {
                Some(br) if br.group >= self.limits.max_bind_groups => {
                    return Err(pipeline::CreateShaderModuleError::InvalidGroupIndex {
                        bind: br,
                        group: br.group,
                        limit: self.limits.max_bind_groups,
                    });
                }
                _ => continue,
            };
        }

        profiling::scope!("naga::validate");
        let debug_source =
            if self.instance_flags.contains(wgt::InstanceFlags::DEBUG) && !source.is_empty() {
                Some(hal::DebugSource {
                    file_name: Cow::Owned(
                        desc.label
                            .as_ref()
                            .map_or("shader".to_string(), |l| l.to_string()),
                    ),
                    source_code: Cow::Owned(source.clone()),
                })
            } else {
                None
            };

        let info = create_validator(
            self.features,
            self.downlevel.flags,
            naga::valid::ValidationFlags::all(),
        )
        .validate(&module)
        .map_err(|inner| {
            pipeline::CreateShaderModuleError::Validation(naga::error::ShaderError {
                source,
                label: desc.label.as_ref().map(|l| l.to_string()),
                inner: Box::new(inner),
            })
        })?;

        let interface = validation::Interface::new(&module, &info, self.limits.clone());
        let hal_shader = hal::ShaderInput::Naga(hal::NagaShader {
            module,
            info,
            debug_source,
        });
        let hal_desc = hal::ShaderModuleDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            runtime_checks: desc.runtime_checks,
        };
        let raw = match unsafe { self.raw().create_shader_module(&hal_desc, hal_shader) } {
            Ok(raw) => raw,
            Err(error) => {
                return Err(match error {
                    hal::ShaderError::Device(error) => {
                        pipeline::CreateShaderModuleError::Device(self.handle_hal_error(error))
                    }
                    hal::ShaderError::Compilation(ref msg) => {
                        log::error!("Shader error: {msg}");
                        pipeline::CreateShaderModuleError::Generation
                    }
                })
            }
        };

        let module = pipeline::ShaderModule {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            interface: Some(interface),
            label: desc.label.to_string(),
        };

        let module = Arc::new(module);

        Ok(module)
    }

    #[allow(unused_unsafe)]
    pub(crate) unsafe fn create_shader_module_passthrough<'a>(
        self: &Arc<Self>,
        descriptor: &pipeline::ShaderModuleDescriptorPassthrough<'a>,
    ) -> Result<Arc<pipeline::ShaderModule>, pipeline::CreateShaderModuleError> {
        self.check_is_valid()?;
        let hal_shader = match descriptor {
            pipeline::ShaderModuleDescriptorPassthrough::SpirV(inner) => {
                self.require_features(wgt::Features::SPIRV_SHADER_PASSTHROUGH)?;
                hal::ShaderInput::SpirV(&inner.source)
            }
            pipeline::ShaderModuleDescriptorPassthrough::Msl(inner) => {
                self.require_features(wgt::Features::MSL_SHADER_PASSTHROUGH)?;
                hal::ShaderInput::Msl {
                    shader: inner.source.to_string(),
                    entry_point: inner.entry_point.to_string(),
                    num_workgroups: inner.num_workgroups,
                }
            }
            pipeline::ShaderModuleDescriptorPassthrough::Dxil(inner) => {
                self.require_features(wgt::Features::HLSL_DXIL_SHADER_PASSTHROUGH)?;
                hal::ShaderInput::Dxil {
                    shader: inner.source,
                    entry_point: inner.entry_point.clone(),
                    num_workgroups: inner.num_workgroups,
                }
            }
            pipeline::ShaderModuleDescriptorPassthrough::Hlsl(inner) => {
                self.require_features(wgt::Features::HLSL_DXIL_SHADER_PASSTHROUGH)?;
                hal::ShaderInput::Hlsl {
                    shader: inner.source,
                    entry_point: inner.entry_point.clone(),
                    num_workgroups: inner.num_workgroups,
                }
            }
        };

        let hal_desc = hal::ShaderModuleDescriptor {
            label: descriptor.label().to_hal(self.instance_flags),
            runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
        };

        let raw = match unsafe { self.raw().create_shader_module(&hal_desc, hal_shader) } {
            Ok(raw) => raw,
            Err(error) => {
                return Err(match error {
                    hal::ShaderError::Device(error) => {
                        pipeline::CreateShaderModuleError::Device(self.handle_hal_error(error))
                    }
                    hal::ShaderError::Compilation(ref msg) => {
                        log::error!("Shader error: {msg}");
                        pipeline::CreateShaderModuleError::Generation
                    }
                })
            }
        };

        let module = pipeline::ShaderModule {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            interface: None,
            label: descriptor.label().to_string(),
        };

        Ok(Arc::new(module))
    }

    pub(crate) fn create_command_encoder(
        self: &Arc<Self>,
        label: &crate::Label,
    ) -> Result<Arc<command::CommandEncoder>, DeviceError> {
        self.check_is_valid()?;

        let queue = self.get_queue().unwrap();

        let encoder = self
            .command_allocator
            .acquire_encoder(self.raw(), queue.raw())
            .map_err(|e| self.handle_hal_error(e))?;

        let cmd_enc = command::CommandEncoder::new(encoder, self, label);

        let cmd_enc = Arc::new(cmd_enc);

        Ok(cmd_enc)
    }

    /// Generate information about late-validated buffer bindings for pipelines.
    //TODO: should this be combined with `get_introspection_bind_group_layouts` in some way?
    fn make_late_sized_buffer_groups(
        shader_binding_sizes: &FastHashMap<naga::ResourceBinding, wgt::BufferSize>,
        layout: &binding_model::PipelineLayout,
    ) -> ArrayVec<pipeline::LateSizedBufferGroup, { hal::MAX_BIND_GROUPS }> {
        // Given the shader-required binding sizes and the pipeline layout,
        // return the filtered list of them in the layout order,
        // removing those with given `min_binding_size`.
        layout
            .bind_group_layouts
            .iter()
            .enumerate()
            .map(|(group_index, bgl)| pipeline::LateSizedBufferGroup {
                shader_sizes: bgl
                    .entries
                    .values()
                    .filter_map(|entry| match entry.ty {
                        wgt::BindingType::Buffer {
                            min_binding_size: None,
                            ..
                        } => {
                            let rb = naga::ResourceBinding {
                                group: group_index as u32,
                                binding: entry.binding,
                            };
                            let shader_size =
                                shader_binding_sizes.get(&rb).map_or(0, |nz| nz.get());
                            Some(shader_size)
                        }
                        _ => None,
                    })
                    .collect(),
            })
            .collect()
    }

    pub(crate) fn create_bind_group_layout(
        self: &Arc<Self>,
        label: &crate::Label,
        entry_map: bgl::EntryMap,
        origin: bgl::Origin,
    ) -> Result<Arc<BindGroupLayout>, binding_model::CreateBindGroupLayoutError> {
        #[derive(PartialEq)]
        enum WritableStorage {
            Yes,
            No,
        }

        for entry in entry_map.values() {
            use wgt::BindingType as Bt;

            let mut required_features = wgt::Features::empty();
            let mut required_downlevel_flags = wgt::DownlevelFlags::empty();
            let (array_feature, writable_storage) = match entry.ty {
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: _,
                } => (
                    Some(wgt::Features::BUFFER_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: _,
                } => (
                    Some(wgt::Features::BUFFER_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only },
                    ..
                } => (
                    Some(
                        wgt::Features::BUFFER_BINDING_ARRAY
                            | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    ),
                    match read_only {
                        true => WritableStorage::No,
                        false => WritableStorage::Yes,
                    },
                ),
                Bt::Sampler { .. } => (
                    Some(wgt::Features::TEXTURE_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::Texture {
                    multisampled: true,
                    sample_type: TextureSampleType::Float { filterable: true },
                    ..
                } => {
                    return Err(binding_model::CreateBindGroupLayoutError::Entry {
                        binding: entry.binding,
                        error:
                            BindGroupLayoutEntryError::SampleTypeFloatFilterableBindingMultisampled,
                    });
                }
                Bt::Texture {
                    multisampled,
                    view_dimension,
                    ..
                } => {
                    if multisampled && view_dimension != TextureViewDimension::D2 {
                        return Err(binding_model::CreateBindGroupLayoutError::Entry {
                            binding: entry.binding,
                            error: BindGroupLayoutEntryError::Non2DMultisampled(view_dimension),
                        });
                    }

                    (
                        Some(wgt::Features::TEXTURE_BINDING_ARRAY),
                        WritableStorage::No,
                    )
                }
                Bt::StorageTexture {
                    access,
                    view_dimension,
                    format: _,
                } => {
                    match view_dimension {
                        TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                            return Err(binding_model::CreateBindGroupLayoutError::Entry {
                                binding: entry.binding,
                                error: BindGroupLayoutEntryError::StorageTextureCube,
                            })
                        }
                        _ => (),
                    }
                    match access {
                        wgt::StorageTextureAccess::Atomic
                            if !self.features.contains(wgt::Features::TEXTURE_ATOMIC) =>
                        {
                            return Err(binding_model::CreateBindGroupLayoutError::Entry {
                                binding: entry.binding,
                                error: BindGroupLayoutEntryError::StorageTextureAtomic,
                            });
                        }
                        _ => (),
                    }
                    (
                        Some(
                            wgt::Features::TEXTURE_BINDING_ARRAY
                                | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                        ),
                        match access {
                            wgt::StorageTextureAccess::WriteOnly => WritableStorage::Yes,
                            wgt::StorageTextureAccess::ReadOnly => WritableStorage::No,
                            wgt::StorageTextureAccess::ReadWrite => WritableStorage::Yes,
                            wgt::StorageTextureAccess::Atomic => {
                                required_features |= wgt::Features::TEXTURE_ATOMIC;
                                WritableStorage::Yes
                            }
                        },
                    )
                }
                Bt::AccelerationStructure { vertex_return } => {
                    self.require_features(wgt::Features::EXPERIMENTAL_RAY_QUERY)
                        .map_err(|e| binding_model::CreateBindGroupLayoutError::Entry {
                            binding: entry.binding,
                            error: e.into(),
                        })?;
                    if vertex_return {
                        self.require_features(wgt::Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)
                            .map_err(|e| binding_model::CreateBindGroupLayoutError::Entry {
                                binding: entry.binding,
                                error: e.into(),
                            })?;
                    }

                    (None, WritableStorage::No)
                }
                Bt::ExternalTexture => {
                    self.require_features(wgt::Features::EXTERNAL_TEXTURE)
                        .map_err(|e| binding_model::CreateBindGroupLayoutError::Entry {
                            binding: entry.binding,
                            error: e.into(),
                        })?;
                    (None, WritableStorage::No)
                }
            };

            // Validate the count parameter
            if entry.count.is_some() {
                required_features |= array_feature
                    .ok_or(BindGroupLayoutEntryError::ArrayUnsupported)
                    .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                        binding: entry.binding,
                        error,
                    })?;
            }

            if entry.visibility.contains_unknown_bits() {
                return Err(
                    binding_model::CreateBindGroupLayoutError::InvalidVisibility(entry.visibility),
                );
            }

            if entry.visibility.contains(wgt::ShaderStages::VERTEX) {
                if writable_storage == WritableStorage::Yes {
                    required_features |= wgt::Features::VERTEX_WRITABLE_STORAGE;
                }
                if let Bt::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    ..
                } = entry.ty
                {
                    required_downlevel_flags |= wgt::DownlevelFlags::VERTEX_STORAGE;
                }
            }
            if writable_storage == WritableStorage::Yes
                && entry.visibility.contains(wgt::ShaderStages::FRAGMENT)
            {
                required_downlevel_flags |= wgt::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE;
            }

            self.require_features(required_features)
                .map_err(BindGroupLayoutEntryError::MissingFeatures)
                .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                    binding: entry.binding,
                    error,
                })?;
            self.require_downlevel_flags(required_downlevel_flags)
                .map_err(BindGroupLayoutEntryError::MissingDownlevelFlags)
                .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                    binding: entry.binding,
                    error,
                })?;
        }

        let bgl_flags = conv::bind_group_layout_flags(self.features);

        let hal_bindings = entry_map.values().copied().collect::<Vec<_>>();
        let hal_desc = hal::BindGroupLayoutDescriptor {
            label: label.to_hal(self.instance_flags),
            flags: bgl_flags,
            entries: &hal_bindings,
        };

        let mut count_validator = binding_model::BindingTypeMaxCountValidator::default();
        for entry in entry_map.values() {
            count_validator.add_binding(entry);
        }
        // If a single bind group layout violates limits, the pipeline layout is
        // definitely going to violate limits too, lets catch it now.
        count_validator
            .validate(&self.limits)
            .map_err(binding_model::CreateBindGroupLayoutError::TooManyBindings)?;

        // Validate that binding arrays don't conflict with dynamic offsets.
        count_validator.validate_binding_arrays()?;

        let raw = unsafe { self.raw().create_bind_group_layout(&hal_desc) }
            .map_err(|e| self.handle_hal_error(e))?;

        let bgl = BindGroupLayout {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            entries: entry_map,
            origin,
            exclusive_pipeline: OnceCellOrLock::new(),
            binding_count_validator: count_validator,
            label: label.to_string(),
        };

        let bgl = Arc::new(bgl);

        Ok(bgl)
    }

    fn create_buffer_binding<'a>(
        &self,
        bb: &'a binding_model::ResolvedBufferBinding,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        used_buffer_ranges: &mut Vec<BufferInitTrackerAction>,
        dynamic_binding_info: &mut Vec<binding_model::BindGroupDynamicBindingData>,
        late_buffer_binding_sizes: &mut FastHashMap<u32, wgt::BufferSize>,
        used: &mut BindGroupStates,
        snatch_guard: &'a SnatchGuard<'a>,
    ) -> Result<hal::BufferBinding<'a, dyn hal::DynBuffer>, binding_model::CreateBindGroupError>
    {
        use crate::binding_model::CreateBindGroupError as Error;

        let (binding_ty, dynamic, min_size) = match decl.ty {
            wgt::BindingType::Buffer {
                ty,
                has_dynamic_offset,
                min_binding_size,
            } => (ty, has_dynamic_offset, min_binding_size),
            _ => {
                return Err(Error::WrongBindingType {
                    binding,
                    actual: decl.ty,
                    expected: "UniformBuffer, StorageBuffer or ReadonlyStorageBuffer",
                })
            }
        };

        let (pub_usage, internal_use, range_limit) = match binding_ty {
            wgt::BufferBindingType::Uniform => (
                wgt::BufferUsages::UNIFORM,
                wgt::BufferUses::UNIFORM,
                self.limits.max_uniform_buffer_binding_size,
            ),
            wgt::BufferBindingType::Storage { read_only } => (
                wgt::BufferUsages::STORAGE,
                if read_only {
                    wgt::BufferUses::STORAGE_READ_ONLY
                } else {
                    wgt::BufferUses::STORAGE_READ_WRITE
                },
                self.limits.max_storage_buffer_binding_size,
            ),
        };

        let (align, align_limit_name) =
            binding_model::buffer_binding_type_alignment(&self.limits, binding_ty);
        if bb.offset % align as u64 != 0 {
            return Err(Error::UnalignedBufferOffset(
                bb.offset,
                align_limit_name,
                align,
            ));
        }

        let buffer = &bb.buffer;

        used.buffers.insert_single(buffer.clone(), internal_use);

        buffer.same_device(self)?;

        buffer.check_usage(pub_usage)?;

        let (bb, bind_size) = buffer.binding(bb.offset, bb.size, snatch_guard)?;

        if matches!(binding_ty, wgt::BufferBindingType::Storage { .. }) {
            let storage_buf_size_alignment = 4;

            let aligned = bind_size % u64::from(storage_buf_size_alignment) == 0;
            if !aligned {
                return Err(Error::UnalignedEffectiveBufferBindingSizeForStorage {
                    alignment: storage_buf_size_alignment,
                    size: bind_size,
                });
            }
        }

        let bind_end = bb.offset + bind_size;

        if bind_size > range_limit as u64 {
            return Err(Error::BufferRangeTooLarge {
                binding,
                given: bind_size as u32,
                limit: range_limit,
            });
        }

        // Record binding info for validating dynamic offsets
        if dynamic {
            dynamic_binding_info.push(binding_model::BindGroupDynamicBindingData {
                binding_idx: binding,
                buffer_size: buffer.size,
                binding_range: bb.offset..bind_end,
                maximum_dynamic_offset: buffer.size - bind_end,
                binding_type: binding_ty,
            });
        }

        if let Some(non_zero) = min_size {
            let min_size = non_zero.get();
            if min_size > bind_size {
                return Err(Error::BindingSizeTooSmall {
                    buffer: buffer.error_ident(),
                    actual: bind_size,
                    min: min_size,
                });
            }
        } else {
            let late_size = wgt::BufferSize::new(bind_size)
                .ok_or_else(|| Error::BindingZeroSize(buffer.error_ident()))?;
            late_buffer_binding_sizes.insert(binding, late_size);
        }

        // This was checked against the device's alignment requirements above,
        // which should always be a multiple of `COPY_BUFFER_ALIGNMENT`.
        assert_eq!(bb.offset % wgt::COPY_BUFFER_ALIGNMENT, 0);

        // `wgpu_hal` only restricts shader access to bound buffer regions with
        // a certain resolution. For the sake of lazy initialization, round up
        // the size of the bound range to reflect how much of the buffer is
        // actually going to be visible to the shader.
        let bounds_check_alignment =
            binding_model::buffer_binding_type_bounds_check_alignment(&self.alignments, binding_ty);
        let visible_size = align_to(bind_size, bounds_check_alignment);

        used_buffer_ranges.extend(buffer.initialization_status.read().create_action(
            buffer,
            bb.offset..bb.offset + visible_size,
            MemoryInitKind::NeedsInitializedMemory,
        ));

        Ok(bb)
    }

    fn create_sampler_binding<'a>(
        &self,
        used: &mut BindGroupStates,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        sampler: &'a Arc<Sampler>,
    ) -> Result<&'a dyn hal::DynSampler, binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;

        used.samplers.insert_single(sampler.clone());

        sampler.same_device(self)?;

        match decl.ty {
            wgt::BindingType::Sampler(ty) => {
                let (allowed_filtering, allowed_comparison) = match ty {
                    wgt::SamplerBindingType::Filtering => (None, false),
                    wgt::SamplerBindingType::NonFiltering => (Some(false), false),
                    wgt::SamplerBindingType::Comparison => (None, true),
                };
                if let Some(allowed_filtering) = allowed_filtering {
                    if allowed_filtering != sampler.filtering {
                        return Err(Error::WrongSamplerFiltering {
                            binding,
                            layout_flt: allowed_filtering,
                            sampler_flt: sampler.filtering,
                        });
                    }
                }
                if allowed_comparison != sampler.comparison {
                    return Err(Error::WrongSamplerComparison {
                        binding,
                        layout_cmp: allowed_comparison,
                        sampler_cmp: sampler.comparison,
                    });
                }
            }
            _ => {
                return Err(Error::WrongBindingType {
                    binding,
                    actual: decl.ty,
                    expected: "Sampler",
                })
            }
        }

        Ok(sampler.raw())
    }

    fn create_texture_binding<'a>(
        &self,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        view: &'a Arc<TextureView>,
        used: &mut BindGroupStates,
        used_texture_ranges: &mut Vec<TextureInitTrackerAction>,
        snatch_guard: &'a SnatchGuard<'a>,
    ) -> Result<hal::TextureBinding<'a, dyn hal::DynTextureView>, binding_model::CreateBindGroupError>
    {
        view.same_device(self)?;

        let internal_use = self.texture_use_parameters(
            binding,
            decl,
            view,
            "SampledTexture, ReadonlyStorageTexture or WriteonlyStorageTexture",
        )?;

        used.views.insert_single(view.clone(), internal_use);

        let texture = &view.parent;

        used_texture_ranges.push(TextureInitTrackerAction {
            texture: texture.clone(),
            range: TextureInitRange {
                mip_range: view.desc.range.mip_range(texture.desc.mip_level_count),
                layer_range: view
                    .desc
                    .range
                    .layer_range(texture.desc.array_layer_count()),
            },
            kind: MemoryInitKind::NeedsInitializedMemory,
        });

        Ok(hal::TextureBinding {
            view: view.try_raw(snatch_guard)?,
            usage: internal_use,
        })
    }

    fn create_tlas_binding<'a>(
        self: &Arc<Self>,
        used: &mut BindGroupStates,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        tlas: &'a Arc<Tlas>,
        snatch_guard: &'a SnatchGuard<'a>,
    ) -> Result<&'a dyn hal::DynAccelerationStructure, binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;

        used.acceleration_structures.insert_single(tlas.clone());

        tlas.same_device(self)?;

        match decl.ty {
            wgt::BindingType::AccelerationStructure { vertex_return } => {
                if vertex_return
                    && !tlas.flags.contains(
                        wgpu_types::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN,
                    )
                {
                    return Err(Error::MissingTLASVertexReturn { binding });
                }
            }
            _ => {
                return Err(Error::WrongBindingType {
                    binding,
                    actual: decl.ty,
                    expected: "Tlas",
                });
            }
        }

        Ok(tlas.try_raw(snatch_guard)?)
    }

    fn create_external_texture_binding<'a>(
        &'a self,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        external_texture: &'a Arc<ExternalTexture>,
        used: &mut BindGroupStates,
        snatch_guard: &'a SnatchGuard,
    ) -> Result<
        hal::ExternalTextureBinding<'a, dyn hal::DynBuffer, dyn hal::DynTextureView>,
        binding_model::CreateBindGroupError,
    > {
        use crate::binding_model::CreateBindGroupError as Error;

        external_texture.same_device(self)?;

        used.external_textures
            .insert_single(external_texture.clone());

        match decl.ty {
            wgt::BindingType::ExternalTexture => {}
            _ => {
                return Err(Error::WrongBindingType {
                    binding,
                    actual: decl.ty,
                    expected: "ExternalTexture",
                });
            }
        }

        let planes = (0..3)
            .map(|i| {
                // We always need 3 bindings. If we have fewer than 3 planes
                // just bind plane 0 multiple times. The shader will only
                // sample from valid planes anyway.
                let plane = external_texture
                    .planes
                    .get(i)
                    .unwrap_or(&external_texture.planes[0]);
                let internal_use = wgt::TextureUses::RESOURCE;
                used.views.insert_single(plane.clone(), internal_use);
                let view = plane.try_raw(snatch_guard)?;
                Ok(hal::TextureBinding {
                    view,
                    usage: internal_use,
                })
            })
            // We can remove this intermediate Vec by using
            // array::try_from_fn() above, once it stabilizes.
            .collect::<Result<Vec<_>, Error>>()?;
        let planes = planes.try_into().unwrap();

        used.buffers
            .insert_single(external_texture.params.clone(), wgt::BufferUses::UNIFORM);
        let params = external_texture.params.binding(0, None, snatch_guard)?.0;

        Ok(hal::ExternalTextureBinding { planes, params })
    }

    fn create_external_texture_binding_from_view<'a>(
        &'a self,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        view: &'a Arc<TextureView>,
        used: &mut BindGroupStates,
        snatch_guard: &'a SnatchGuard,
    ) -> Result<
        hal::ExternalTextureBinding<'a, dyn hal::DynBuffer, dyn hal::DynTextureView>,
        binding_model::CreateBindGroupError,
    > {
        use crate::binding_model::CreateBindGroupError as Error;

        view.same_device(self)?;

        let internal_use = self.texture_use_parameters(binding, decl, view, "SampledTexture")?;
        used.views.insert_single(view.clone(), internal_use);

        match decl.ty {
            wgt::BindingType::ExternalTexture => {}
            _ => {
                return Err(Error::WrongBindingType {
                    binding,
                    actual: decl.ty,
                    expected: "ExternalTexture",
                });
            }
        }

        // We need 3 bindings, so just repeat the same texture view 3 times.
        let planes = [
            hal::TextureBinding {
                view: view.try_raw(snatch_guard)?,
                usage: internal_use,
            },
            hal::TextureBinding {
                view: view.try_raw(snatch_guard)?,
                usage: internal_use,
            },
            hal::TextureBinding {
                view: view.try_raw(snatch_guard)?,
                usage: internal_use,
            },
        ];
        let params = hal::BufferBinding::new_unchecked(
            self.default_external_texture_params_buffer.as_ref(),
            0,
            None,
        );

        Ok(hal::ExternalTextureBinding { planes, params })
    }

    // This function expects the provided bind group layout to be resolved
    // (not passing a duplicate) beforehand.
    pub(crate) fn create_bind_group(
        self: &Arc<Self>,
        desc: binding_model::ResolvedBindGroupDescriptor,
    ) -> Result<Arc<BindGroup>, binding_model::CreateBindGroupError> {
        use crate::binding_model::{CreateBindGroupError as Error, ResolvedBindingResource as Br};

        let layout = desc.layout;

        self.check_is_valid()?;
        layout.same_device(self)?;

        {
            // Check that the number of entries in the descriptor matches
            // the number of entries in the layout.
            let actual = desc.entries.len();
            let expected = layout.entries.len();
            if actual != expected {
                return Err(Error::BindingsNumMismatch { expected, actual });
            }
        }

        // TODO: arrayvec/smallvec, or re-use allocations
        // Record binding info for dynamic offset validation
        let mut dynamic_binding_info = Vec::new();
        // Map of binding -> shader reflected size
        //Note: we can't collect into a vector right away because
        // it needs to be in BGL iteration order, not BG entry order.
        let mut late_buffer_binding_sizes = FastHashMap::default();
        // fill out the descriptors
        let mut used = BindGroupStates::new();

        let mut used_buffer_ranges = Vec::new();
        let mut used_texture_ranges = Vec::new();
        let mut hal_entries = Vec::with_capacity(desc.entries.len());
        let mut hal_buffers = Vec::new();
        let mut hal_samplers = Vec::new();
        let mut hal_textures = Vec::new();
        let mut hal_tlas_s = Vec::new();
        let mut hal_external_textures = Vec::new();
        let snatch_guard = self.snatchable_lock.read();
        for entry in desc.entries.iter() {
            let binding = entry.binding;
            // Find the corresponding declaration in the layout
            let decl = layout
                .entries
                .get(binding)
                .ok_or(Error::MissingBindingDeclaration(binding))?;
            let (res_index, count) = match entry.resource {
                Br::Buffer(ref bb) => {
                    let bb = self.create_buffer_binding(
                        bb,
                        binding,
                        decl,
                        &mut used_buffer_ranges,
                        &mut dynamic_binding_info,
                        &mut late_buffer_binding_sizes,
                        &mut used,
                        &snatch_guard,
                    )?;

                    let res_index = hal_buffers.len();
                    hal_buffers.push(bb);
                    (res_index, 1)
                }
                Br::BufferArray(ref bindings_array) => {
                    let num_bindings = bindings_array.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_buffers.len();
                    for bb in bindings_array.iter() {
                        let bb = self.create_buffer_binding(
                            bb,
                            binding,
                            decl,
                            &mut used_buffer_ranges,
                            &mut dynamic_binding_info,
                            &mut late_buffer_binding_sizes,
                            &mut used,
                            &snatch_guard,
                        )?;
                        hal_buffers.push(bb);
                    }
                    (res_index, num_bindings)
                }
                Br::Sampler(ref sampler) => {
                    let sampler = self.create_sampler_binding(&mut used, binding, decl, sampler)?;

                    let res_index = hal_samplers.len();
                    hal_samplers.push(sampler);
                    (res_index, 1)
                }
                Br::SamplerArray(ref samplers) => {
                    let num_bindings = samplers.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_samplers.len();
                    for sampler in samplers.iter() {
                        let sampler =
                            self.create_sampler_binding(&mut used, binding, decl, sampler)?;

                        hal_samplers.push(sampler);
                    }

                    (res_index, num_bindings)
                }
                Br::TextureView(ref view) => match decl.ty {
                    wgt::BindingType::ExternalTexture => {
                        let et = self.create_external_texture_binding_from_view(
                            binding,
                            decl,
                            view,
                            &mut used,
                            &snatch_guard,
                        )?;
                        let res_index = hal_external_textures.len();
                        hal_external_textures.push(et);
                        (res_index, 1)
                    }
                    _ => {
                        let tb = self.create_texture_binding(
                            binding,
                            decl,
                            view,
                            &mut used,
                            &mut used_texture_ranges,
                            &snatch_guard,
                        )?;
                        let res_index = hal_textures.len();
                        hal_textures.push(tb);
                        (res_index, 1)
                    }
                },
                Br::TextureViewArray(ref views) => {
                    let num_bindings = views.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_textures.len();
                    for view in views.iter() {
                        let tb = self.create_texture_binding(
                            binding,
                            decl,
                            view,
                            &mut used,
                            &mut used_texture_ranges,
                            &snatch_guard,
                        )?;

                        hal_textures.push(tb);
                    }

                    (res_index, num_bindings)
                }
                Br::AccelerationStructure(ref tlas) => {
                    let tlas =
                        self.create_tlas_binding(&mut used, binding, decl, tlas, &snatch_guard)?;
                    let res_index = hal_tlas_s.len();
                    hal_tlas_s.push(tlas);
                    (res_index, 1)
                }
                Br::ExternalTexture(ref et) => {
                    let et = self.create_external_texture_binding(
                        binding,
                        decl,
                        et,
                        &mut used,
                        &snatch_guard,
                    )?;
                    let res_index = hal_external_textures.len();
                    hal_external_textures.push(et);
                    (res_index, 1)
                }
            };

            hal_entries.push(hal::BindGroupEntry {
                binding,
                resource_index: res_index as u32,
                count: count as u32,
            });
        }

        used.optimize();

        hal_entries.sort_by_key(|entry| entry.binding);
        for (a, b) in hal_entries.iter().zip(hal_entries.iter().skip(1)) {
            if a.binding == b.binding {
                return Err(Error::DuplicateBinding(a.binding));
            }
        }
        let hal_desc = hal::BindGroupDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: layout.raw(),
            entries: &hal_entries,
            buffers: &hal_buffers,
            samplers: &hal_samplers,
            textures: &hal_textures,
            acceleration_structures: &hal_tlas_s,
            external_textures: &hal_external_textures,
        };
        let raw = unsafe { self.raw().create_bind_group(&hal_desc) }
            .map_err(|e| self.handle_hal_error(e))?;

        // collect in the order of BGL iteration
        let late_buffer_binding_sizes = layout
            .entries
            .indices()
            .flat_map(|binding| late_buffer_binding_sizes.get(&binding).cloned())
            .collect();

        let bind_group = BindGroup {
            raw: Snatchable::new(raw),
            device: self.clone(),
            layout,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.bind_groups.clone()),
            used,
            used_buffer_ranges,
            used_texture_ranges,
            dynamic_binding_info,
            late_buffer_binding_sizes,
        };

        let bind_group = Arc::new(bind_group);

        let weak_ref = Arc::downgrade(&bind_group);
        for range in &bind_group.used_texture_ranges {
            let mut bind_groups = range.texture.bind_groups.lock();
            bind_groups.push(weak_ref.clone());
        }
        for range in &bind_group.used_buffer_ranges {
            let mut bind_groups = range.buffer.bind_groups.lock();
            bind_groups.push(weak_ref.clone());
        }

        Ok(bind_group)
    }

    fn check_array_binding(
        features: wgt::Features,
        count: Option<NonZeroU32>,
        num_bindings: usize,
    ) -> Result<(), binding_model::CreateBindGroupError> {
        use super::binding_model::CreateBindGroupError as Error;

        if let Some(count) = count {
            let count = count.get() as usize;
            if count < num_bindings {
                return Err(Error::BindingArrayPartialLengthMismatch {
                    actual: num_bindings,
                    expected: count,
                });
            }
            if count != num_bindings
                && !features.contains(wgt::Features::PARTIALLY_BOUND_BINDING_ARRAY)
            {
                return Err(Error::BindingArrayLengthMismatch {
                    actual: num_bindings,
                    expected: count,
                });
            }
            if num_bindings == 0 {
                return Err(Error::BindingArrayZeroLength);
            }
        } else {
            return Err(Error::SingleBindingExpected);
        };

        Ok(())
    }

    fn texture_use_parameters(
        &self,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        view: &TextureView,
        expected: &'static str,
    ) -> Result<wgt::TextureUses, binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;
        if view
            .desc
            .aspects()
            .contains(hal::FormatAspects::DEPTH | hal::FormatAspects::STENCIL)
        {
            return Err(Error::DepthStencilAspect);
        }
        match decl.ty {
            wgt::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled,
            } => {
                use wgt::TextureSampleType as Tst;
                if multisampled != (view.samples != 1) {
                    return Err(Error::InvalidTextureMultisample {
                        binding,
                        layout_multisampled: multisampled,
                        view_samples: view.samples,
                    });
                }
                let compat_sample_type = view
                    .desc
                    .format
                    .sample_type(Some(view.desc.range.aspect), Some(self.features))
                    .unwrap();
                match (sample_type, compat_sample_type) {
                    (Tst::Uint, Tst::Uint) |
                        (Tst::Sint, Tst::Sint) |
                        (Tst::Depth, Tst::Depth) |
                        // if we expect non-filterable, accept anything float
                        (Tst::Float { filterable: false }, Tst::Float { .. }) |
                        // if we expect filterable, require it
                        (Tst::Float { filterable: true }, Tst::Float { filterable: true }) |
                        // if we expect non-filterable, also accept depth
                        (Tst::Float { filterable: false }, Tst::Depth) => {}
                    // if we expect filterable, also accept Float that is defined as
                    // unfilterable if filterable feature is explicitly enabled (only hit
                    // if wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES is
                    // enabled)
                    (Tst::Float { filterable: true }, Tst::Float { .. }) if view.format_features.flags.contains(wgt::TextureFormatFeatureFlags::FILTERABLE) => {}
                    _ => {
                        return Err(Error::InvalidTextureSampleType {
                            binding,
                            layout_sample_type: sample_type,
                            view_format: view.desc.format,
                            view_sample_type: compat_sample_type,
                        })
                    }
                }
                if view_dimension != view.desc.dimension {
                    return Err(Error::InvalidTextureDimension {
                        binding,
                        layout_dimension: view_dimension,
                        view_dimension: view.desc.dimension,
                    });
                }
                view.check_usage(wgt::TextureUsages::TEXTURE_BINDING)?;
                Ok(wgt::TextureUses::RESOURCE)
            }
            wgt::BindingType::StorageTexture {
                access,
                format,
                view_dimension,
            } => {
                if format != view.desc.format {
                    return Err(Error::InvalidStorageTextureFormat {
                        binding,
                        layout_format: format,
                        view_format: view.desc.format,
                    });
                }
                if view_dimension != view.desc.dimension {
                    return Err(Error::InvalidTextureDimension {
                        binding,
                        layout_dimension: view_dimension,
                        view_dimension: view.desc.dimension,
                    });
                }

                let mip_level_count = view.selector.mips.end - view.selector.mips.start;
                if mip_level_count != 1 {
                    return Err(Error::InvalidStorageTextureMipLevelCount {
                        binding,
                        mip_level_count,
                    });
                }

                let internal_use = match access {
                    wgt::StorageTextureAccess::WriteOnly => {
                        if !view
                            .format_features
                            .flags
                            .contains(wgt::TextureFormatFeatureFlags::STORAGE_WRITE_ONLY)
                        {
                            return Err(Error::StorageWriteNotSupported(view.desc.format));
                        }
                        wgt::TextureUses::STORAGE_WRITE_ONLY
                    }
                    wgt::StorageTextureAccess::ReadOnly => {
                        if !view
                            .format_features
                            .flags
                            .contains(wgt::TextureFormatFeatureFlags::STORAGE_READ_ONLY)
                        {
                            return Err(Error::StorageReadNotSupported(view.desc.format));
                        }
                        wgt::TextureUses::STORAGE_READ_ONLY
                    }
                    wgt::StorageTextureAccess::ReadWrite => {
                        if !view
                            .format_features
                            .flags
                            .contains(wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE)
                        {
                            return Err(Error::StorageReadWriteNotSupported(view.desc.format));
                        }

                        wgt::TextureUses::STORAGE_READ_WRITE
                    }
                    wgt::StorageTextureAccess::Atomic => {
                        if !view
                            .format_features
                            .flags
                            .contains(wgt::TextureFormatFeatureFlags::STORAGE_ATOMIC)
                        {
                            return Err(Error::StorageAtomicNotSupported(view.desc.format));
                        }

                        wgt::TextureUses::STORAGE_ATOMIC
                    }
                };
                view.check_usage(wgt::TextureUsages::STORAGE_BINDING)?;
                Ok(internal_use)
            }
            wgt::BindingType::ExternalTexture => {
                if view.desc.dimension != TextureViewDimension::D2 {
                    return Err(Error::InvalidTextureDimension {
                        binding,
                        layout_dimension: TextureViewDimension::D2,
                        view_dimension: view.desc.dimension,
                    });
                }
                let mip_level_count = view.selector.mips.end - view.selector.mips.start;
                if mip_level_count != 1 {
                    return Err(Error::InvalidExternalTextureMipLevelCount {
                        binding,
                        mip_level_count,
                    });
                }
                if view.desc.format != TextureFormat::Rgba8Unorm
                    && view.desc.format != TextureFormat::Bgra8Unorm
                    && view.desc.format != TextureFormat::Rgba16Float
                {
                    return Err(Error::InvalidExternalTextureFormat {
                        binding,
                        format: view.desc.format,
                    });
                }
                if view.samples != 1 {
                    return Err(Error::InvalidTextureMultisample {
                        binding,
                        layout_multisampled: false,
                        view_samples: view.samples,
                    });
                }

                view.check_usage(wgt::TextureUsages::TEXTURE_BINDING)?;
                Ok(wgt::TextureUses::RESOURCE)
            }
            _ => Err(Error::WrongBindingType {
                binding,
                actual: decl.ty,
                expected,
            }),
        }
    }

    pub(crate) fn create_pipeline_layout(
        self: &Arc<Self>,
        desc: &binding_model::ResolvedPipelineLayoutDescriptor,
    ) -> Result<Arc<binding_model::PipelineLayout>, binding_model::CreatePipelineLayoutError> {
        use crate::binding_model::CreatePipelineLayoutError as Error;

        self.check_is_valid()?;

        let bind_group_layouts_count = desc.bind_group_layouts.len();
        let device_max_bind_groups = self.limits.max_bind_groups as usize;
        if bind_group_layouts_count > device_max_bind_groups {
            return Err(Error::TooManyGroups {
                actual: bind_group_layouts_count,
                max: device_max_bind_groups,
            });
        }

        if !desc.push_constant_ranges.is_empty() {
            self.require_features(wgt::Features::PUSH_CONSTANTS)?;
        }

        let mut used_stages = wgt::ShaderStages::empty();
        for (index, pc) in desc.push_constant_ranges.iter().enumerate() {
            if pc.stages.intersects(used_stages) {
                return Err(Error::MoreThanOnePushConstantRangePerStage {
                    index,
                    provided: pc.stages,
                    intersected: pc.stages & used_stages,
                });
            }
            used_stages |= pc.stages;

            let device_max_pc_size = self.limits.max_push_constant_size;
            if device_max_pc_size < pc.range.end {
                return Err(Error::PushConstantRangeTooLarge {
                    index,
                    range: pc.range.clone(),
                    max: device_max_pc_size,
                });
            }

            if pc.range.start % wgt::PUSH_CONSTANT_ALIGNMENT != 0 {
                return Err(Error::MisalignedPushConstantRange {
                    index,
                    bound: pc.range.start,
                });
            }
            if pc.range.end % wgt::PUSH_CONSTANT_ALIGNMENT != 0 {
                return Err(Error::MisalignedPushConstantRange {
                    index,
                    bound: pc.range.end,
                });
            }
        }

        let mut count_validator = binding_model::BindingTypeMaxCountValidator::default();

        for bgl in desc.bind_group_layouts.iter() {
            bgl.same_device(self)?;
            count_validator.merge(&bgl.binding_count_validator);
        }

        count_validator
            .validate(&self.limits)
            .map_err(Error::TooManyBindings)?;

        let bind_group_layouts = desc
            .bind_group_layouts
            .iter()
            .cloned()
            .collect::<ArrayVec<_, { hal::MAX_BIND_GROUPS }>>();

        let raw_bind_group_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.raw())
            .collect::<ArrayVec<_, { hal::MAX_BIND_GROUPS }>>();

        let additional_flags = if self.indirect_validation.is_some() {
            hal::PipelineLayoutFlags::INDIRECT_BUILTIN_UPDATE
        } else {
            hal::PipelineLayoutFlags::empty()
        };

        let hal_desc = hal::PipelineLayoutDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            flags: hal::PipelineLayoutFlags::FIRST_VERTEX_INSTANCE
                | hal::PipelineLayoutFlags::NUM_WORK_GROUPS
                | additional_flags,
            bind_group_layouts: &raw_bind_group_layouts,
            push_constant_ranges: desc.push_constant_ranges.as_ref(),
        };

        let raw = unsafe { self.raw().create_pipeline_layout(&hal_desc) }
            .map_err(|e| self.handle_hal_error(e))?;

        drop(raw_bind_group_layouts);

        let layout = binding_model::PipelineLayout {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            label: desc.label.to_string(),
            bind_group_layouts,
            push_constant_ranges: desc.push_constant_ranges.iter().cloned().collect(),
        };

        let layout = Arc::new(layout);

        Ok(layout)
    }

    pub(crate) fn derive_pipeline_layout(
        self: &Arc<Self>,
        mut derived_group_layouts: Box<ArrayVec<bgl::EntryMap, { hal::MAX_BIND_GROUPS }>>,
    ) -> Result<Arc<binding_model::PipelineLayout>, pipeline::ImplicitLayoutError> {
        while derived_group_layouts
            .last()
            .is_some_and(|map| map.is_empty())
        {
            derived_group_layouts.pop();
        }

        let mut unique_bind_group_layouts = FastHashMap::default();

        let bind_group_layouts = derived_group_layouts
            .into_iter()
            .map(|mut bgl_entry_map| {
                bgl_entry_map.sort();
                match unique_bind_group_layouts.entry(bgl_entry_map) {
                    hashbrown::hash_map::Entry::Occupied(v) => Ok(Arc::clone(v.get())),
                    hashbrown::hash_map::Entry::Vacant(e) => {
                        match self.create_bind_group_layout(
                            &None,
                            e.key().clone(),
                            bgl::Origin::Derived,
                        ) {
                            Ok(bgl) => {
                                e.insert(bgl.clone());
                                Ok(bgl)
                            }
                            Err(e) => Err(e),
                        }
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let layout_desc = binding_model::ResolvedPipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: Cow::Owned(bind_group_layouts),
            push_constant_ranges: Cow::Borrowed(&[]), //TODO?
        };

        let layout = self.create_pipeline_layout(&layout_desc)?;
        Ok(layout)
    }

    pub(crate) fn create_compute_pipeline(
        self: &Arc<Self>,
        desc: pipeline::ResolvedComputePipelineDescriptor,
    ) -> Result<Arc<pipeline::ComputePipeline>, pipeline::CreateComputePipelineError> {
        self.check_is_valid()?;

        self.require_downlevel_flags(wgt::DownlevelFlags::COMPUTE_SHADERS)?;

        let shader_module = desc.stage.module;

        shader_module.same_device(self)?;

        let is_auto_layout = desc.layout.is_none();

        // Get the pipeline layout from the desc if it is provided.
        let pipeline_layout = match desc.layout {
            Some(pipeline_layout) => {
                pipeline_layout.same_device(self)?;
                Some(pipeline_layout)
            }
            None => None,
        };

        let mut binding_layout_source = match pipeline_layout {
            Some(ref pipeline_layout) => {
                validation::BindingLayoutSource::Provided(pipeline_layout.get_binding_maps())
            }
            None => validation::BindingLayoutSource::new_derived(&self.limits),
        };
        let mut shader_binding_sizes = FastHashMap::default();
        let io = validation::StageIo::default();

        let final_entry_point_name;

        {
            let stage = wgt::ShaderStages::COMPUTE;

            final_entry_point_name = shader_module.finalize_entry_point_name(
                stage,
                desc.stage.entry_point.as_ref().map(|ep| ep.as_ref()),
            )?;

            if let Some(ref interface) = shader_module.interface {
                let _ = interface.check_stage(
                    &mut binding_layout_source,
                    &mut shader_binding_sizes,
                    &final_entry_point_name,
                    stage,
                    io,
                    None,
                )?;
            }
        }

        let pipeline_layout = match binding_layout_source {
            validation::BindingLayoutSource::Provided(_) => {
                drop(binding_layout_source);
                pipeline_layout.unwrap()
            }
            validation::BindingLayoutSource::Derived(entries) => {
                self.derive_pipeline_layout(entries)?
            }
        };

        let late_sized_buffer_groups =
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, &pipeline_layout);

        let cache = match desc.cache {
            Some(cache) => {
                cache.same_device(self)?;
                Some(cache)
            }
            None => None,
        };

        let pipeline_desc = hal::ComputePipelineDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: pipeline_layout.raw(),
            stage: hal::ProgrammableStage {
                module: shader_module.raw(),
                entry_point: final_entry_point_name.as_ref(),
                constants: &desc.stage.constants,
                zero_initialize_workgroup_memory: desc.stage.zero_initialize_workgroup_memory,
            },
            cache: cache.as_ref().map(|it| it.raw()),
        };

        let raw =
            unsafe { self.raw().create_compute_pipeline(&pipeline_desc) }.map_err(
                |err| match err {
                    hal::PipelineError::Device(error) => {
                        pipeline::CreateComputePipelineError::Device(self.handle_hal_error(error))
                    }
                    hal::PipelineError::Linkage(_stages, msg) => {
                        pipeline::CreateComputePipelineError::Internal(msg)
                    }
                    hal::PipelineError::EntryPoint(_stage) => {
                        pipeline::CreateComputePipelineError::Internal(
                            ENTRYPOINT_FAILURE_ERROR.to_string(),
                        )
                    }
                    hal::PipelineError::PipelineConstants(_stages, msg) => {
                        pipeline::CreateComputePipelineError::PipelineConstants(msg)
                    }
                },
            )?;

        let pipeline = pipeline::ComputePipeline {
            raw: ManuallyDrop::new(raw),
            layout: pipeline_layout,
            device: self.clone(),
            _shader_module: shader_module,
            late_sized_buffer_groups,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.compute_pipelines.clone()),
        };

        let pipeline = Arc::new(pipeline);

        if is_auto_layout {
            for bgl in pipeline.layout.bind_group_layouts.iter() {
                // `bind_group_layouts` might contain duplicate entries, so we need to ignore the result.
                let _ = bgl
                    .exclusive_pipeline
                    .set(binding_model::ExclusivePipeline::Compute(Arc::downgrade(
                        &pipeline,
                    )));
            }
        }

        Ok(pipeline)
    }

    pub(crate) fn create_render_pipeline(
        self: &Arc<Self>,
        desc: pipeline::ResolvedGeneralRenderPipelineDescriptor,
    ) -> Result<Arc<pipeline::RenderPipeline>, pipeline::CreateRenderPipelineError> {
        use wgt::TextureFormatFeatureFlags as Tfff;

        self.check_is_valid()?;

        let mut shader_binding_sizes = FastHashMap::default();

        let num_attachments = desc.fragment.as_ref().map(|f| f.targets.len()).unwrap_or(0);
        let max_attachments = self.limits.max_color_attachments as usize;
        if num_attachments > max_attachments {
            return Err(pipeline::CreateRenderPipelineError::ColorAttachment(
                command::ColorAttachmentError::TooMany {
                    given: num_attachments,
                    limit: max_attachments,
                },
            ));
        }

        let color_targets = desc
            .fragment
            .as_ref()
            .map_or(&[][..], |fragment| &fragment.targets);
        let depth_stencil_state = desc.depth_stencil.as_ref();

        {
            let cts: ArrayVec<_, { hal::MAX_COLOR_ATTACHMENTS }> =
                color_targets.iter().filter_map(|x| x.as_ref()).collect();
            if !cts.is_empty() && {
                let first = &cts[0];
                cts[1..]
                    .iter()
                    .any(|ct| ct.write_mask != first.write_mask || ct.blend != first.blend)
            } {
                self.require_downlevel_flags(wgt::DownlevelFlags::INDEPENDENT_BLEND)?;
            }
        }

        let mut io = validation::StageIo::default();
        let mut validated_stages = wgt::ShaderStages::empty();

        let mut vertex_steps;
        let mut vertex_buffers;
        let mut total_attributes;
        let mut shader_expects_dual_source_blending = false;
        let mut pipeline_expects_dual_source_blending = false;
        if let pipeline::RenderPipelineVertexProcessor::Vertex(ref vertex) = desc.vertex {
            vertex_steps = Vec::with_capacity(vertex.buffers.len());
            vertex_buffers = Vec::with_capacity(vertex.buffers.len());
            total_attributes = 0;
            shader_expects_dual_source_blending = false;
            pipeline_expects_dual_source_blending = false;
            for (i, vb_state) in vertex.buffers.iter().enumerate() {
                // https://gpuweb.github.io/gpuweb/#abstract-opdef-validating-gpuvertexbufferlayout

                if vb_state.array_stride > self.limits.max_vertex_buffer_array_stride as u64 {
                    return Err(pipeline::CreateRenderPipelineError::VertexStrideTooLarge {
                        index: i as u32,
                        given: vb_state.array_stride as u32,
                        limit: self.limits.max_vertex_buffer_array_stride,
                    });
                }
                if vb_state.array_stride % wgt::VERTEX_ALIGNMENT != 0 {
                    return Err(pipeline::CreateRenderPipelineError::UnalignedVertexStride {
                        index: i as u32,
                        stride: vb_state.array_stride,
                    });
                }

                let max_stride = if vb_state.array_stride == 0 {
                    self.limits.max_vertex_buffer_array_stride as u64
                } else {
                    vb_state.array_stride
                };
                let mut last_stride = 0;
                for attribute in vb_state.attributes.iter() {
                    let attribute_stride = attribute.offset + attribute.format.size();
                    if attribute_stride > max_stride {
                        return Err(
                            pipeline::CreateRenderPipelineError::VertexAttributeStrideTooLarge {
                                location: attribute.shader_location,
                                given: attribute_stride as u32,
                                limit: max_stride as u32,
                            },
                        );
                    }

                    let required_offset_alignment = attribute.format.size().min(4);
                    if attribute.offset % required_offset_alignment != 0 {
                        return Err(
                            pipeline::CreateRenderPipelineError::InvalidVertexAttributeOffset {
                                location: attribute.shader_location,
                                offset: attribute.offset,
                            },
                        );
                    }

                    if attribute.shader_location >= self.limits.max_vertex_attributes {
                        return Err(
                            pipeline::CreateRenderPipelineError::TooManyVertexAttributes {
                                given: attribute.shader_location,
                                limit: self.limits.max_vertex_attributes,
                            },
                        );
                    }

                    last_stride = last_stride.max(attribute_stride);
                }
                vertex_steps.push(pipeline::VertexStep {
                    stride: vb_state.array_stride,
                    last_stride,
                    mode: vb_state.step_mode,
                });
                if vb_state.attributes.is_empty() {
                    continue;
                }
                vertex_buffers.push(hal::VertexBufferLayout {
                    array_stride: vb_state.array_stride,
                    step_mode: vb_state.step_mode,
                    attributes: vb_state.attributes.as_ref(),
                });

                for attribute in vb_state.attributes.iter() {
                    if attribute.offset >= 0x10000000 {
                        return Err(
                            pipeline::CreateRenderPipelineError::InvalidVertexAttributeOffset {
                                location: attribute.shader_location,
                                offset: attribute.offset,
                            },
                        );
                    }

                    if let wgt::VertexFormat::Float64
                    | wgt::VertexFormat::Float64x2
                    | wgt::VertexFormat::Float64x3
                    | wgt::VertexFormat::Float64x4 = attribute.format
                    {
                        self.require_features(wgt::Features::VERTEX_ATTRIBUTE_64BIT)?;
                    }

                    let previous = io.insert(
                        attribute.shader_location,
                        validation::InterfaceVar::vertex_attribute(attribute.format),
                    );

                    if previous.is_some() {
                        return Err(pipeline::CreateRenderPipelineError::ShaderLocationClash(
                            attribute.shader_location,
                        ));
                    }
                }
                total_attributes += vb_state.attributes.len();
            }

            if vertex_buffers.len() > self.limits.max_vertex_buffers as usize {
                return Err(pipeline::CreateRenderPipelineError::TooManyVertexBuffers {
                    given: vertex_buffers.len() as u32,
                    limit: self.limits.max_vertex_buffers,
                });
            }
            if total_attributes > self.limits.max_vertex_attributes as usize {
                return Err(
                    pipeline::CreateRenderPipelineError::TooManyVertexAttributes {
                        given: total_attributes as u32,
                        limit: self.limits.max_vertex_attributes,
                    },
                );
            }
        } else {
            vertex_steps = Vec::new();
            vertex_buffers = Vec::new();
        };

        if desc.primitive.strip_index_format.is_some() && !desc.primitive.topology.is_strip() {
            return Err(
                pipeline::CreateRenderPipelineError::StripIndexFormatForNonStripTopology {
                    strip_index_format: desc.primitive.strip_index_format,
                    topology: desc.primitive.topology,
                },
            );
        }

        if desc.primitive.unclipped_depth {
            self.require_features(wgt::Features::DEPTH_CLIP_CONTROL)?;
        }

        if desc.primitive.polygon_mode == wgt::PolygonMode::Line {
            self.require_features(wgt::Features::POLYGON_MODE_LINE)?;
        }
        if desc.primitive.polygon_mode == wgt::PolygonMode::Point {
            self.require_features(wgt::Features::POLYGON_MODE_POINT)?;
        }

        if desc.primitive.conservative {
            self.require_features(wgt::Features::CONSERVATIVE_RASTERIZATION)?;
        }

        if desc.primitive.conservative && desc.primitive.polygon_mode != wgt::PolygonMode::Fill {
            return Err(
                pipeline::CreateRenderPipelineError::ConservativeRasterizationNonFillPolygonMode,
            );
        }

        let mut target_specified = false;

        for (i, cs) in color_targets.iter().enumerate() {
            if let Some(cs) = cs.as_ref() {
                target_specified = true;
                let error = 'error: {
                    if cs.write_mask.contains_unknown_bits() {
                        break 'error Some(pipeline::ColorStateError::InvalidWriteMask(
                            cs.write_mask,
                        ));
                    }

                    let format_features = self.describe_format_features(cs.format)?;
                    if !format_features
                        .allowed_usages
                        .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                    {
                        break 'error Some(pipeline::ColorStateError::FormatNotRenderable(
                            cs.format,
                        ));
                    }
                    let blendable = format_features.flags.contains(Tfff::BLENDABLE);
                    let filterable = format_features.flags.contains(Tfff::FILTERABLE);
                    let adapter_specific = self
                        .features
                        .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
                    // according to WebGPU specifications the texture needs to be
                    // [`TextureFormatFeatureFlags::FILTERABLE`] if blending is set - use
                    // [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] to elude
                    // this limitation
                    if cs.blend.is_some() && (!blendable || (!filterable && !adapter_specific)) {
                        break 'error Some(pipeline::ColorStateError::FormatNotBlendable(
                            cs.format,
                        ));
                    }
                    if !hal::FormatAspects::from(cs.format).contains(hal::FormatAspects::COLOR) {
                        break 'error Some(pipeline::ColorStateError::FormatNotColor(cs.format));
                    }

                    if desc.multisample.count > 1
                        && !format_features
                            .flags
                            .sample_count_supported(desc.multisample.count)
                    {
                        break 'error Some(pipeline::ColorStateError::InvalidSampleCount(
                            desc.multisample.count,
                            cs.format,
                            cs.format
                                .guaranteed_format_features(self.features)
                                .flags
                                .supported_sample_counts(),
                            self.adapter
                                .get_texture_format_features(cs.format)
                                .flags
                                .supported_sample_counts(),
                        ));
                    }

                    if let Some(blend_mode) = cs.blend {
                        for factor in [
                            blend_mode.color.src_factor,
                            blend_mode.color.dst_factor,
                            blend_mode.alpha.src_factor,
                            blend_mode.alpha.dst_factor,
                        ] {
                            if factor.ref_second_blend_source() {
                                self.require_features(wgt::Features::DUAL_SOURCE_BLENDING)?;
                                if i == 0 {
                                    pipeline_expects_dual_source_blending = true;
                                    break;
                                } else {
                                    return Err(pipeline::CreateRenderPipelineError
                                        ::BlendFactorOnUnsupportedTarget { factor, target: i as u32 });
                                }
                            }
                        }
                    }

                    break 'error None;
                };
                if let Some(e) = error {
                    return Err(pipeline::CreateRenderPipelineError::ColorState(i as u8, e));
                }
            }
        }

        let limit = self.limits.max_color_attachment_bytes_per_sample;
        let formats = color_targets
            .iter()
            .map(|cs| cs.as_ref().map(|cs| cs.format));
        if let Err(total) = validate_color_attachment_bytes_per_sample(formats, limit) {
            return Err(pipeline::CreateRenderPipelineError::ColorAttachment(
                command::ColorAttachmentError::TooManyBytesPerSample { total, limit },
            ));
        }

        if let Some(ds) = depth_stencil_state {
            target_specified = true;
            let error = 'error: {
                let format_features = self.describe_format_features(ds.format)?;
                if !format_features
                    .allowed_usages
                    .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                {
                    break 'error Some(pipeline::DepthStencilStateError::FormatNotRenderable(
                        ds.format,
                    ));
                }

                let aspect = hal::FormatAspects::from(ds.format);
                if ds.is_depth_enabled() && !aspect.contains(hal::FormatAspects::DEPTH) {
                    break 'error Some(pipeline::DepthStencilStateError::FormatNotDepth(ds.format));
                }
                if ds.stencil.is_enabled() && !aspect.contains(hal::FormatAspects::STENCIL) {
                    break 'error Some(pipeline::DepthStencilStateError::FormatNotStencil(
                        ds.format,
                    ));
                }
                if desc.multisample.count > 1
                    && !format_features
                        .flags
                        .sample_count_supported(desc.multisample.count)
                {
                    break 'error Some(pipeline::DepthStencilStateError::InvalidSampleCount(
                        desc.multisample.count,
                        ds.format,
                        ds.format
                            .guaranteed_format_features(self.features)
                            .flags
                            .supported_sample_counts(),
                        self.adapter
                            .get_texture_format_features(ds.format)
                            .flags
                            .supported_sample_counts(),
                    ));
                }

                break 'error None;
            };
            if let Some(e) = error {
                return Err(pipeline::CreateRenderPipelineError::DepthStencilState(e));
            }

            if ds.bias.clamp != 0.0 {
                self.require_downlevel_flags(wgt::DownlevelFlags::DEPTH_BIAS_CLAMP)?;
            }
        }

        if !target_specified {
            return Err(pipeline::CreateRenderPipelineError::NoTargetSpecified);
        }

        let is_auto_layout = desc.layout.is_none();

        // Get the pipeline layout from the desc if it is provided.
        let pipeline_layout = match desc.layout {
            Some(pipeline_layout) => {
                pipeline_layout.same_device(self)?;
                Some(pipeline_layout)
            }
            None => None,
        };

        let mut binding_layout_source = match pipeline_layout {
            Some(ref pipeline_layout) => {
                validation::BindingLayoutSource::Provided(pipeline_layout.get_binding_maps())
            }
            None => validation::BindingLayoutSource::new_derived(&self.limits),
        };

        let samples = {
            let sc = desc.multisample.count;
            if sc == 0 || sc > 32 || !sc.is_power_of_two() {
                return Err(pipeline::CreateRenderPipelineError::InvalidSampleCount(sc));
            }
            sc
        };

        let mut vertex_stage = None;
        let mut task_stage = None;
        let mut mesh_stage = None;
        let mut _vertex_entry_point_name = String::new();
        let mut _task_entry_point_name = String::new();
        let mut _mesh_entry_point_name = String::new();
        match desc.vertex {
            pipeline::RenderPipelineVertexProcessor::Vertex(ref vertex) => {
                vertex_stage = {
                    let stage_desc = &vertex.stage;
                    let stage = wgt::ShaderStages::VERTEX;

                    let vertex_shader_module = &stage_desc.module;
                    vertex_shader_module.same_device(self)?;

                    let stage_err =
                        |error| pipeline::CreateRenderPipelineError::Stage { stage, error };

                    _vertex_entry_point_name = vertex_shader_module
                        .finalize_entry_point_name(
                            stage,
                            stage_desc.entry_point.as_ref().map(|ep| ep.as_ref()),
                        )
                        .map_err(stage_err)?;

                    if let Some(ref interface) = vertex_shader_module.interface {
                        io = interface
                            .check_stage(
                                &mut binding_layout_source,
                                &mut shader_binding_sizes,
                                &_vertex_entry_point_name,
                                stage,
                                io,
                                desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                            )
                            .map_err(stage_err)?;
                        validated_stages |= stage;
                    }
                    Some(hal::ProgrammableStage {
                        module: vertex_shader_module.raw(),
                        entry_point: &_vertex_entry_point_name,
                        constants: &stage_desc.constants,
                        zero_initialize_workgroup_memory: stage_desc
                            .zero_initialize_workgroup_memory,
                    })
                };
            }
            pipeline::RenderPipelineVertexProcessor::Mesh(ref task, ref mesh) => {
                task_stage = if let Some(task) = task {
                    let stage_desc = &task.stage;
                    let stage = wgt::ShaderStages::TASK;
                    let task_shader_module = &stage_desc.module;
                    task_shader_module.same_device(self)?;

                    let stage_err =
                        |error| pipeline::CreateRenderPipelineError::Stage { stage, error };

                    _task_entry_point_name = task_shader_module
                        .finalize_entry_point_name(
                            stage,
                            stage_desc.entry_point.as_ref().map(|ep| ep.as_ref()),
                        )
                        .map_err(stage_err)?;

                    if let Some(ref interface) = task_shader_module.interface {
                        io = interface
                            .check_stage(
                                &mut binding_layout_source,
                                &mut shader_binding_sizes,
                                &_task_entry_point_name,
                                stage,
                                io,
                                desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                            )
                            .map_err(stage_err)?;
                        validated_stages |= stage;
                    }
                    Some(hal::ProgrammableStage {
                        module: task_shader_module.raw(),
                        entry_point: &_task_entry_point_name,
                        constants: &stage_desc.constants,
                        zero_initialize_workgroup_memory: stage_desc
                            .zero_initialize_workgroup_memory,
                    })
                } else {
                    None
                };
                mesh_stage = {
                    let stage_desc = &mesh.stage;
                    let stage = wgt::ShaderStages::MESH;
                    let mesh_shader_module = &stage_desc.module;
                    mesh_shader_module.same_device(self)?;

                    let stage_err =
                        |error| pipeline::CreateRenderPipelineError::Stage { stage, error };

                    _mesh_entry_point_name = mesh_shader_module
                        .finalize_entry_point_name(
                            stage,
                            stage_desc.entry_point.as_ref().map(|ep| ep.as_ref()),
                        )
                        .map_err(stage_err)?;

                    if let Some(ref interface) = mesh_shader_module.interface {
                        io = interface
                            .check_stage(
                                &mut binding_layout_source,
                                &mut shader_binding_sizes,
                                &_mesh_entry_point_name,
                                stage,
                                io,
                                desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                            )
                            .map_err(stage_err)?;
                        validated_stages |= stage;
                    }
                    Some(hal::ProgrammableStage {
                        module: mesh_shader_module.raw(),
                        entry_point: &_mesh_entry_point_name,
                        constants: &stage_desc.constants,
                        zero_initialize_workgroup_memory: stage_desc
                            .zero_initialize_workgroup_memory,
                    })
                };
            }
        }

        let fragment_entry_point_name;
        let fragment_stage = match desc.fragment {
            Some(ref fragment_state) => {
                let stage = wgt::ShaderStages::FRAGMENT;

                let shader_module = &fragment_state.stage.module;
                shader_module.same_device(self)?;

                let stage_err = |error| pipeline::CreateRenderPipelineError::Stage { stage, error };

                fragment_entry_point_name = shader_module
                    .finalize_entry_point_name(
                        stage,
                        fragment_state
                            .stage
                            .entry_point
                            .as_ref()
                            .map(|ep| ep.as_ref()),
                    )
                    .map_err(stage_err)?;

                if validated_stages == wgt::ShaderStages::VERTEX {
                    if let Some(ref interface) = shader_module.interface {
                        io = interface
                            .check_stage(
                                &mut binding_layout_source,
                                &mut shader_binding_sizes,
                                &fragment_entry_point_name,
                                stage,
                                io,
                                desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                            )
                            .map_err(stage_err)?;
                        validated_stages |= stage;
                    }
                }

                if let Some(ref interface) = shader_module.interface {
                    shader_expects_dual_source_blending = interface
                        .fragment_uses_dual_source_blending(&fragment_entry_point_name)
                        .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                            stage,
                            error,
                        })?;
                }

                Some(hal::ProgrammableStage {
                    module: shader_module.raw(),
                    entry_point: &fragment_entry_point_name,
                    constants: &fragment_state.stage.constants,
                    zero_initialize_workgroup_memory: fragment_state
                        .stage
                        .zero_initialize_workgroup_memory,
                })
            }
            None => None,
        };

        if !pipeline_expects_dual_source_blending && shader_expects_dual_source_blending {
            return Err(
                pipeline::CreateRenderPipelineError::ShaderExpectsPipelineToUseDualSourceBlending,
            );
        }
        if pipeline_expects_dual_source_blending && !shader_expects_dual_source_blending {
            return Err(
                pipeline::CreateRenderPipelineError::PipelineExpectsShaderToUseDualSourceBlending,
            );
        }

        if validated_stages.contains(wgt::ShaderStages::FRAGMENT) {
            for (i, output) in io.iter() {
                match color_targets.get(*i as usize) {
                    Some(Some(state)) => {
                        validation::check_texture_format(state.format, &output.ty).map_err(
                            |pipeline| {
                                pipeline::CreateRenderPipelineError::ColorState(
                                    *i as u8,
                                    pipeline::ColorStateError::IncompatibleFormat {
                                        pipeline,
                                        shader: output.ty,
                                    },
                                )
                            },
                        )?;
                    }
                    _ => {
                        log::warn!(
                            "The fragment stage {:?} output @location({}) values are ignored",
                            fragment_stage
                                .as_ref()
                                .map_or("", |stage| stage.entry_point),
                            i
                        );
                    }
                }
            }
        }
        let last_stage = match desc.fragment {
            Some(_) => wgt::ShaderStages::FRAGMENT,
            None => wgt::ShaderStages::VERTEX,
        };
        if is_auto_layout && !validated_stages.contains(last_stage) {
            return Err(pipeline::ImplicitLayoutError::ReflectionError(last_stage).into());
        }

        let pipeline_layout = match binding_layout_source {
            validation::BindingLayoutSource::Provided(_) => {
                drop(binding_layout_source);
                pipeline_layout.unwrap()
            }
            validation::BindingLayoutSource::Derived(entries) => {
                self.derive_pipeline_layout(entries)?
            }
        };

        // Multiview is only supported if the feature is enabled
        if desc.multiview.is_some() {
            self.require_features(wgt::Features::MULTIVIEW)?;
        }

        if !self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED)
        {
            for (binding, size) in shader_binding_sizes.iter() {
                if size.get() % 16 != 0 {
                    return Err(pipeline::CreateRenderPipelineError::UnalignedShader {
                        binding: binding.binding,
                        group: binding.group,
                        size: size.get(),
                    });
                }
            }
        }

        let late_sized_buffer_groups =
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, &pipeline_layout);

        let cache = match desc.cache {
            Some(cache) => {
                cache.same_device(self)?;
                Some(cache)
            }
            None => None,
        };

        let is_mesh = mesh_stage.is_some();
        let raw = {
            let pipeline_desc = hal::RenderPipelineDescriptor {
                label: desc.label.to_hal(self.instance_flags),
                layout: pipeline_layout.raw(),
                vertex_processor: match vertex_stage {
                    Some(vertex_stage) => hal::VertexProcessor::Standard {
                        vertex_buffers: &vertex_buffers,
                        vertex_stage,
                    },
                    None => hal::VertexProcessor::Mesh {
                        task_stage,
                        mesh_stage: mesh_stage.unwrap(),
                    },
                },
                primitive: desc.primitive,
                depth_stencil: desc.depth_stencil.clone(),
                multisample: desc.multisample,
                fragment_stage,
                color_targets,
                multiview: desc.multiview,
                cache: cache.as_ref().map(|it| it.raw()),
            };
            unsafe { self.raw().create_render_pipeline(&pipeline_desc) }.map_err(
                |err| match err {
                    hal::PipelineError::Device(error) => {
                        pipeline::CreateRenderPipelineError::Device(self.handle_hal_error(error))
                    }
                    hal::PipelineError::Linkage(stage, msg) => {
                        pipeline::CreateRenderPipelineError::Internal { stage, error: msg }
                    }
                    hal::PipelineError::EntryPoint(stage) => {
                        pipeline::CreateRenderPipelineError::Internal {
                            stage: hal::auxil::map_naga_stage(stage),
                            error: ENTRYPOINT_FAILURE_ERROR.to_string(),
                        }
                    }
                    hal::PipelineError::PipelineConstants(stage, error) => {
                        pipeline::CreateRenderPipelineError::PipelineConstants { stage, error }
                    }
                },
            )?
        };

        let pass_context = RenderPassContext {
            attachments: AttachmentData {
                colors: color_targets
                    .iter()
                    .map(|state| state.as_ref().map(|s| s.format))
                    .collect(),
                resolves: ArrayVec::new(),
                depth_stencil: depth_stencil_state.as_ref().map(|state| state.format),
            },
            sample_count: samples,
            multiview: desc.multiview,
        };

        let mut flags = pipeline::PipelineFlags::empty();
        for state in color_targets.iter().filter_map(|s| s.as_ref()) {
            if let Some(ref bs) = state.blend {
                if bs.color.uses_constant() | bs.alpha.uses_constant() {
                    flags |= pipeline::PipelineFlags::BLEND_CONSTANT;
                }
            }
        }
        if let Some(ds) = depth_stencil_state.as_ref() {
            if ds.stencil.is_enabled() && ds.stencil.needs_ref_value() {
                flags |= pipeline::PipelineFlags::STENCIL_REFERENCE;
            }
            if !ds.is_depth_read_only() {
                flags |= pipeline::PipelineFlags::WRITES_DEPTH;
            }
            if !ds.is_stencil_read_only(desc.primitive.cull_mode) {
                flags |= pipeline::PipelineFlags::WRITES_STENCIL;
            }
        }
        let shader_modules = {
            let mut shader_modules = ArrayVec::new();
            match desc.vertex {
                pipeline::RenderPipelineVertexProcessor::Vertex(vertex) => {
                    shader_modules.push(vertex.stage.module)
                }
                pipeline::RenderPipelineVertexProcessor::Mesh(task, mesh) => {
                    if let Some(task) = task {
                        shader_modules.push(task.stage.module);
                    }
                    shader_modules.push(mesh.stage.module);
                }
            }
            shader_modules.extend(desc.fragment.map(|f| f.stage.module));
            shader_modules
        };

        let pipeline = pipeline::RenderPipeline {
            raw: ManuallyDrop::new(raw),
            layout: pipeline_layout,
            device: self.clone(),
            pass_context,
            _shader_modules: shader_modules,
            flags,
            strip_index_format: desc.primitive.strip_index_format,
            vertex_steps,
            late_sized_buffer_groups,
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.render_pipelines.clone()),
            is_mesh,
        };

        let pipeline = Arc::new(pipeline);

        if is_auto_layout {
            for bgl in pipeline.layout.bind_group_layouts.iter() {
                // `bind_group_layouts` might contain duplicate entries, so we need to ignore the result.
                let _ = bgl
                    .exclusive_pipeline
                    .set(binding_model::ExclusivePipeline::Render(Arc::downgrade(
                        &pipeline,
                    )));
            }
        }

        Ok(pipeline)
    }

    /// # Safety
    /// The `data` field on `desc` must have previously been returned from [`crate::global::Global::pipeline_cache_get_data`]
    pub unsafe fn create_pipeline_cache(
        self: &Arc<Self>,
        desc: &pipeline::PipelineCacheDescriptor,
    ) -> Result<Arc<pipeline::PipelineCache>, pipeline::CreatePipelineCacheError> {
        use crate::pipeline_cache;

        self.check_is_valid()?;

        self.require_features(wgt::Features::PIPELINE_CACHE)?;
        let data = if let Some((data, validation_key)) = desc
            .data
            .as_ref()
            .zip(self.raw().pipeline_cache_validation_key())
        {
            let data = pipeline_cache::validate_pipeline_cache(
                data,
                &self.adapter.raw.info,
                validation_key,
            );
            match data {
                Ok(data) => Some(data),
                Err(e) if e.was_avoidable() || !desc.fallback => return Err(e.into()),
                // If the error was unavoidable and we are asked to fallback, do so
                Err(_) => None,
            }
        } else {
            None
        };
        let cache_desc = hal::PipelineCacheDescriptor {
            data,
            label: desc.label.to_hal(self.instance_flags),
        };
        let raw = match unsafe { self.raw().create_pipeline_cache(&cache_desc) } {
            Ok(raw) => raw,
            Err(e) => match e {
                hal::PipelineCacheError::Device(e) => return Err(self.handle_hal_error(e).into()),
            },
        };
        let cache = pipeline::PipelineCache {
            device: self.clone(),
            label: desc.label.to_string(),
            // This would be none in the error condition, which we don't implement yet
            raw: ManuallyDrop::new(raw),
        };

        let cache = Arc::new(cache);

        Ok(cache)
    }

    fn get_texture_format_features(&self, format: TextureFormat) -> wgt::TextureFormatFeatures {
        // Variant of adapter.get_texture_format_features that takes device features into account
        use wgt::TextureFormatFeatureFlags as tfsc;
        let mut format_features = self.adapter.get_texture_format_features(format);
        if (format == TextureFormat::R32Float
            || format == TextureFormat::Rg32Float
            || format == TextureFormat::Rgba32Float)
            && !self.features.contains(wgt::Features::FLOAT32_FILTERABLE)
        {
            format_features.flags.set(tfsc::FILTERABLE, false);
        }
        format_features
    }

    fn describe_format_features(
        &self,
        format: TextureFormat,
    ) -> Result<wgt::TextureFormatFeatures, MissingFeatures> {
        self.require_features(format.required_features())?;

        let using_device_features = self
            .features
            .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        // If we're running downlevel, we need to manually ask the backend what
        // we can use as we can't trust WebGPU.
        let downlevel = !self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT);

        if using_device_features || downlevel {
            Ok(self.get_texture_format_features(format))
        } else {
            Ok(format.guaranteed_format_features(self.features))
        }
    }

    #[cfg(feature = "replay")]
    pub(crate) fn wait_for_submit(
        &self,
        submission_index: crate::SubmissionIndex,
    ) -> Result<(), DeviceError> {
        let fence = self.fence.read();
        let last_done_index = unsafe { self.raw().get_fence_value(fence.as_ref()) }
            .map_err(|e| self.handle_hal_error(e))?;
        if last_done_index < submission_index {
            unsafe { self.raw().wait(fence.as_ref(), submission_index, !0) }
                .map_err(|e| self.handle_hal_error(e))?;
            drop(fence);
            if let Some(queue) = self.get_queue() {
                let closures = queue.lock_life().triage_submissions(submission_index);
                assert!(
                    closures.is_empty(),
                    "wait_for_submit is not expected to work with closures"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn create_query_set(
        self: &Arc<Self>,
        desc: &resource::QuerySetDescriptor,
    ) -> Result<Arc<QuerySet>, resource::CreateQuerySetError> {
        use resource::CreateQuerySetError as Error;

        self.check_is_valid()?;

        match desc.ty {
            wgt::QueryType::Occlusion => {}
            wgt::QueryType::Timestamp => {
                self.require_features(wgt::Features::TIMESTAMP_QUERY)?;
            }
            wgt::QueryType::PipelineStatistics(..) => {
                self.require_features(wgt::Features::PIPELINE_STATISTICS_QUERY)?;
            }
        }

        if desc.count == 0 {
            return Err(Error::ZeroCount);
        }

        if desc.count > wgt::QUERY_SET_MAX_QUERIES {
            return Err(Error::TooManyQueries {
                count: desc.count,
                maximum: wgt::QUERY_SET_MAX_QUERIES,
            });
        }

        let hal_desc = desc.map_label(|label| label.to_hal(self.instance_flags));

        let raw = unsafe { self.raw().create_query_set(&hal_desc) }
            .map_err(|e| self.handle_hal_error_with_nonfatal_oom(e))?;

        let query_set = QuerySet {
            raw: ManuallyDrop::new(raw),
            device: self.clone(),
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(self.tracker_indices.query_sets.clone()),
            desc: desc.map_label(|_| ()),
        };

        let query_set = Arc::new(query_set);

        Ok(query_set)
    }

    fn lose(&self, message: &str) {
        // Follow the steps at https://gpuweb.github.io/gpuweb/#lose-the-device.

        // Mark the device explicitly as invalid. This is checked in various
        // places to prevent new work from being submitted.
        self.valid.store(false, Ordering::Release);

        // 1) Resolve the GPUDevice device.lost promise.
        if let Some(device_lost_closure) = self.device_lost_closure.lock().take() {
            device_lost_closure(DeviceLostReason::Unknown, message.to_string());
        }

        // 2) Complete any outstanding mapAsync() steps.
        // 3) Complete any outstanding onSubmittedWorkDone() steps.

        // These parts are passively accomplished by setting valid to false,
        // since that will prevent any new work from being added to the queues.
        // Future calls to poll_devices will continue to check the work queues
        // until they are cleared, and then drop the device.
    }

    fn release_gpu_resources(&self) {
        // This is called when the device is lost, which makes every associated
        // resource invalid and unusable. This is an opportunity to release all of
        // the underlying gpu resources, even though the objects remain visible to
        // the user agent. We purge this memory naturally when resources have been
        // moved into the appropriate buckets, so this function just needs to
        // initiate movement into those buckets, and it can do that by calling
        // "destroy" on all the resources we know about.

        // During these iterations, we discard all errors. We don't care!
        let trackers = self.trackers.lock();
        for buffer in trackers.buffers.used_resources() {
            if let Some(buffer) = Weak::upgrade(buffer) {
                buffer.destroy();
            }
        }
        for texture in trackers.textures.used_resources() {
            if let Some(texture) = Weak::upgrade(texture) {
                texture.destroy();
            }
        }
    }

    pub(crate) fn new_usage_scope(&self) -> UsageScope<'_> {
        UsageScope::new_pooled(&self.usage_scopes, &self.tracker_indices)
    }

    pub fn get_hal_counters(&self) -> wgt::HalCounters {
        self.raw().get_internal_counters()
    }

    pub fn generate_allocator_report(&self) -> Option<wgt::AllocatorReport> {
        self.raw().generate_allocator_report()
    }
}

crate::impl_resource_type!(Device);
crate::impl_labeled!(Device);
crate::impl_storage_item!(Device);
