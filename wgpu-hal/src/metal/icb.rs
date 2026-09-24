//! Lowering of fixed-count multi-draws to Metal indirect command buffers.
//!
//! A multi-draw records the execution of an ICB into the current render
//! pass immediately; the compute work that fills the ICB from the indirect
//! argument buffer is deferred to [`encode_deferred_icb_generation`], which
//! wgpu-core schedules ahead of the pass so the pass is never split.
//!
//! [`encode_deferred_icb_generation`]: super::CommandEncoder::encode_deferred_icb_generation

use super::command::WORD_SIZE;
use alloc::{
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};
use core::ptr::NonNull;
use wgpu_sync::OnceCell;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLArgumentEncoder, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFunction, MTLIndexType,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandBufferExecutionRange, MTLIndirectCommandType, MTLLibrary,
    MTLMeshRenderPipelineDescriptor, MTLPipelineOption, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLRenderStages, MTLResource,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
};

/// Minimum `draw_count` for which lowering a fixed-count multi-draw to an
/// indirect command buffer pays off.
///
/// Measured with a paced frame loop (three frames in flight, ICBs from the
/// pool) on A10X, A12, A14, A18 Pro and M4 Max: below this the per-draw
/// indirect loop is cheaper on both the CPU and the GPU on every one of them,
/// and from here the ICB path's CPU cost stays flat while the loop's grows
/// with every draw. The CPU crossover sits between 256 and 1024 draws
/// depending on the device.
const ICB_MIN_DRAW_COUNT: u32 = 512;

/// `maxVertexBufferBindCount` for ICB descriptors that inherit the encoder's
/// buffers.
///
/// Metal's documentation makes the count irrelevant when buffers are
/// inherited, and Apple3, Apple7, Apple8 and Apple9 GPUs (A10X, A14, A18 Pro,
/// M4 Max) do accept 0. An A12 (Apple5, iOS 18.7) does not: with a count of
/// 0, 16 or 30 -- anything that leaves out a vertex-buffer slot the inherited
/// pipeline reads -- executing the ICB fails its command buffer with a GPU
/// address fault (`kIOGPUCommandBufferCallbackErrorPageFault`). No validation
/// layer catches it and this backend does not surface command-buffer errors,
/// so the pass's draws silently vanish; 31 is healthy. wgpu binds vertex
/// buffers from the top of the 31-slot argument table, so only the full table
/// covers every layout.
const ICB_MAX_INHERITED_BUFFER_BIND_COUNT: usize = 31;

/// Upper bound on the memory one indirect command buffer may take, and so on
/// the draw count the ICB path accepts: larger multi-draws take the per-draw
/// loop. Metal's feature tables list no ICB size limit, but every command is
/// sized for the full inherited state, measured with `allocatedSize`:
///
/// | GPU                         | draw   | indexed |
/// |-----------------------------|--------|---------|
/// | A10X (Apple3, tvOS 26.6)    | 1477 B | 1489 B  |
/// | A12 (Apple5, iOS 18.7)      | 1609 B | 1621 B  |
/// | A14 (Apple7, iOS 26.5)      |  656 B |  672 B  |
/// | M4 Max (Apple9, macOS 27)   |  673 B |  693 B  |
///
/// (mesh-task commands cost 1277 B on the M4 Max), so 64 MiB holds about 40K
/// commands on Apple3 and Apple5 and 100K on Apple7 and later. The bound is
/// checked before allocating because the driver does not fail gracefully at
/// its own limit: an allocation past ~4 GiB returns nil on iOS and tvOS but
/// crashes the process on macOS 27.
const ICB_MAX_BYTES: u64 = 64 << 20;

/// Bounds on the per-adapter pool of indirect command buffers: entries kept,
/// and the total memory they may retain.
const ICB_POOL_MAX_ENTRIES: usize = 8;
const ICB_POOL_MAX_BYTES: u64 = 128 << 20;

// Primitive-topology tags passed to the ICB generation kernels.
// `render_command` in MSL needs the topology per draw, and `MTLPrimitiveType`
// isn't guaranteed stable as an ABI, so we define our own values; they must
// match the `WgpuIcbPrimitiveType` enum in `shaders/icb_generation.metal`.
const ICB_PRIMITIVE_POINT: u32 = 0;
const ICB_PRIMITIVE_LINE: u32 = 1;
const ICB_PRIMITIVE_LINE_STRIP: u32 = 2;
const ICB_PRIMITIVE_TRIANGLE: u32 = 3;
const ICB_PRIMITIVE_TRIANGLE_STRIP: u32 = 4;

const ICB_GENERATION_SHADER: &str = include_str!("./shaders/icb_generation.metal");

const ICB_MESH_GENERATION_SHADER: &str = include_str!("./shaders/icb_mesh_generation.metal");

/// Compute pipelines that translate indirect-draw argument sequences into ICB
/// commands, compiled lazily once per adapter and shared by all command
/// encoders (see [`super::AdapterShared::icb_command_pipelines`]).
#[derive(Clone, Debug)]
pub(super) struct IcbCommandPipelines {
    draw: IcbCommandPipeline,
    indexed_u16: IcbCommandPipeline,
    indexed_u32: IcbCommandPipeline,
    /// Clamps a GPU-resident draw count into an ICB execution range.
    execution_range: IcbCommandPipeline,
    /// Clamps/zeroes indirect args for count draws that can't use an ICB.
    clamp: IcbCommandPipeline,
    /// Present when the adapter reports mesh ICB support and the kernel
    /// compiled; see `PrivateCapabilities::indirect_command_buffers_mesh`.
    mesh: Option<IcbCommandPipeline>,
}

#[derive(Clone, Debug)]
struct IcbCommandPipeline {
    function: Retained<ProtocolObject<dyn MTLFunction>>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

struct IcbArgumentEncoderState {
    encoder: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
    encoded_length: usize,
}

impl IcbArgumentEncoderState {
    fn new(pipeline: &IcbCommandPipeline) -> Self {
        let encoder = unsafe { pipeline.function.newArgumentEncoderWithBufferIndex(0) };
        let encoded_length = encoder.encodedLength();
        Self {
            encoder,
            encoded_length,
        }
    }
}

#[derive(Default)]
pub(super) struct IcbArgumentEncoderCache {
    draw: Option<IcbArgumentEncoderState>,
    indexed_u16: Option<IcbArgumentEncoderState>,
    indexed_u32: Option<IcbArgumentEncoderState>,
    mesh: Option<IcbArgumentEncoderState>,
}

impl IcbArgumentEncoderCache {
    // MTLArgumentEncoder mutates its bound argument buffer state. Cache it per
    // command encoder to avoid per multi-draw allocation without sharing that
    // mutable state between command encoders.
    fn draw(&mut self, pipeline: &IcbCommandPipeline) -> &IcbArgumentEncoderState {
        self.draw
            .get_or_insert_with(|| IcbArgumentEncoderState::new(pipeline))
    }

    fn indexed_u16(&mut self, pipeline: &IcbCommandPipeline) -> &IcbArgumentEncoderState {
        self.indexed_u16
            .get_or_insert_with(|| IcbArgumentEncoderState::new(pipeline))
    }

    fn indexed_u32(&mut self, pipeline: &IcbCommandPipeline) -> &IcbArgumentEncoderState {
        self.indexed_u32
            .get_or_insert_with(|| IcbArgumentEncoderState::new(pipeline))
    }

    fn mesh(&mut self, pipeline: &IcbCommandPipeline) -> &IcbArgumentEncoderState {
        self.mesh
            .get_or_insert_with(|| IcbArgumentEncoderState::new(pipeline))
    }
}

impl IcbCommandPipelines {
    fn make_pipeline_from_library(
        shared: &super::AdapterShared,
        library: &ProtocolObject<dyn MTLLibrary>,
        name: &str,
    ) -> Result<IcbCommandPipeline, crate::DeviceError> {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| {
                log::error!("Metal ICB generation function '{name}' was not found");
                crate::DeviceError::Unexpected
            })?;
        let pipeline = shared
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| {
                log::error!("failed to create Metal ICB generation pipeline '{name}': {err}");
                crate::DeviceError::Unexpected
            })?;

        Ok(IcbCommandPipeline { function, pipeline })
    }

    fn make_library(
        shared: &super::AdapterShared,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, crate::DeviceError> {
        super::device::compile_msl_library(
            &shared.device,
            shared.private_caps.msl_version,
            false,
            source,
        )
        .map_err(|err| {
            log::error!("failed to compile Metal ICB generation shader: {err}");
            crate::DeviceError::Unexpected
        })
    }

    fn new(shared: &super::AdapterShared) -> Result<Self, crate::DeviceError> {
        let library = Self::make_library(shared, ICB_GENERATION_SHADER)?;

        // The mesh kernel is built alongside the draw kernels whenever the
        // adapter reports mesh ICB support. If it fails only the mesh path is
        // lost (the helpers have logged why); the draw kernels stay usable.
        let mesh = if shared.private_caps.indirect_command_buffers_mesh {
            Self::make_library(shared, ICB_MESH_GENERATION_SHADER)
                .and_then(|library| {
                    Self::make_pipeline_from_library(shared, &library, "wgpu_generate_mesh_mdi_icb")
                })
                .ok()
        } else {
            None
        };

        Ok(Self {
            draw: Self::make_pipeline_from_library(shared, &library, "wgpu_generate_mdi_icb")?,
            indexed_u16: Self::make_pipeline_from_library(
                shared,
                &library,
                "wgpu_generate_indexed_mdi_icb_u16",
            )?,
            indexed_u32: Self::make_pipeline_from_library(
                shared,
                &library,
                "wgpu_generate_indexed_mdi_icb_u32",
            )?,
            execution_range: Self::make_pipeline_from_library(
                shared,
                &library,
                "wgpu_generate_mdi_execution_range",
            )?,
            clamp: Self::make_pipeline_from_library(shared, &library, "wgpu_clamp_mdi_args")?,
            mesh,
        })
    }
}

/// Which ICB generation kernel a deferred multi-draw needs, along with the
/// draw-time state that kernel consumes.
pub(super) enum IcbDrawKind {
    Draw,
    DrawIndexed {
        index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        index_offset: wgt::BufferAddress,
        raw_index_type: MTLIndexType,
    },
    DrawMeshTasks {
        /// Object (task) and mesh threadgroup sizes of the pipeline bound at
        /// draw time, `[object_x, object_y, object_z, mesh_x, mesh_y, mesh_z]`.
        threadgroup_sizes: [u32; 6],
    },
}

/// A multi-draw whose ICB execution has been recorded into the render pass but
/// whose generation compute has not been encoded yet; drained by
/// `encode_deferred_multi_draws`.
pub(super) struct IcbGenerationRequest {
    kind: IcbDrawKind,
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    /// Argument buffer through which the generation kernel addresses the ICB,
    /// already encoded at draw time.
    argument_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Buffer holding the packed indirect draw arguments.
    args_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    args_offset: wgt::BufferAddress,
    draw_count: u32,
    /// One of the `ICB_PRIMITIVE_*` values; unused for mesh draws.
    primitive_type_value: u32,
    kind_tag: u8,
    /// Command capacity the ICB was created with; at least `draw_count`.
    capacity: u32,
    /// Memory the ICB and its argument buffer occupy.
    bytes: u64,
}

/// One deferred piece of pre-pass work queued by a multi-draw.
pub(super) enum DeferredMultiDraw {
    /// Fill an ICB that executes with a fixed, CPU-known draw count.
    Icb(IcbGenerationRequest),
    /// Fill an ICB that executes with a GPU-computed range clamped from a
    /// count buffer (`multi_draw_*_indirect_count`).
    IcbCount {
        request: IcbGenerationRequest,
        count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        count_offset: wgt::BufferAddress,
        /// Private buffer receiving the `MTLIndirectCommandBufferExecutionRange`.
        range_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    },
    /// Clamp/zero indirect args into a private buffer that a fixed-length
    /// per-draw loop consumes: the count-draw fallback for pipelines that
    /// can't execute inside an ICB, and for counts past the ICB memory bound.
    ClampedArgs {
        dst_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        src_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        src_offset: wgt::BufferAddress,
        count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        count_offset: wgt::BufferAddress,
        max_count: u32,
        words_per_draw: u32,
    },
}

impl DeferredMultiDraw {
    fn icb_request(&self) -> Option<&IcbGenerationRequest> {
        match self {
            DeferredMultiDraw::Icb(request) | DeferredMultiDraw::IcbCount { request, .. } => {
                Some(request)
            }
            DeferredMultiDraw::ClampedArgs { .. } => None,
        }
    }
}

impl IcbDrawKind {
    /// Which ICB descriptor (command type and bind counts) a draw kind needs;
    /// ICBs are only ever reused for the same tag.
    fn tag(&self) -> u8 {
        match self {
            IcbDrawKind::Draw => 0,
            IcbDrawKind::DrawIndexed { .. } => 1,
            IcbDrawKind::DrawMeshTasks { .. } => 2,
        }
    }
}

/// An indirect command buffer and the argument buffer through which the
/// generation kernels address it, kept for reuse once the command buffer that
/// executed it has completed.
///
/// Creating an ICB is the dominant CPU cost of the whole lowering (hundreds of
/// microseconds for a few thousand commands, growing with the count), so ICBs
/// are recycled: a multi-draw takes the smallest pooled ICB of its kind that
/// holds its draws, and the command buffer that executes it puts it back when
/// it completes.
#[derive(Debug)]
pub(super) struct PooledIcb {
    kind_tag: u8,
    capacity: u32,
    bytes: u64,
    icb: Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>,
    argument_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

#[cfg(send_sync)]
unsafe impl Send for PooledIcb {}
#[cfg(send_sync)]
unsafe impl Sync for PooledIcb {}

/// Objects a submitted command buffer must keep alive; see
/// [`super::CommandBuffer::_icb_resources`]. Dropped when that command buffer
/// has completed, which is when the ICB can safely return to the pool.
pub(super) struct IcbExecutionResources {
    shared: Arc<super::AdapterShared>,
    kind_tag: u8,
    capacity: u32,
    bytes: u64,
    icb: Option<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    argument_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Other buffers the executed commands read: an execution range, or
    /// clamped arguments. Freed with the command buffer, never pooled.
    _extra: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl core::fmt::Debug for IcbExecutionResources {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IcbExecutionResources")
            .field("kind_tag", &self.kind_tag)
            .field("capacity", &self.capacity)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for IcbExecutionResources {
    fn drop(&mut self) {
        let (Some(icb), Some(argument_buffer)) = (self.icb.take(), self.argument_buffer.take())
        else {
            return;
        };
        if self.bytes > ICB_POOL_MAX_BYTES {
            return;
        }
        let mut pool = self.shared.icb_pool.lock();
        let mut total: u64 = pool.iter().map(|entry| entry.bytes).sum();
        while pool.len() >= ICB_POOL_MAX_ENTRIES || total + self.bytes > ICB_POOL_MAX_BYTES {
            // Make room by evicting the smallest entry, unless this one is
            // smaller still: large ICBs are the expensive ones to recreate.
            let Some((index, smallest)) = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.bytes)
                .map(|(index, entry)| (index, entry.bytes))
            else {
                break;
            };
            if smallest >= self.bytes {
                return;
            }
            total -= smallest;
            pool.swap_remove(index);
        }
        pool.push(PooledIcb {
            kind_tag: self.kind_tag,
            capacity: self.capacity,
            bytes: self.bytes,
            icb,
            argument_buffer,
        });
    }
}

/// Descriptor of a render pipeline, kept so the pipeline's ICB-capable twin
/// can be compiled on first use.
pub(super) enum IcbPipelineDescriptor {
    Standard(Retained<MTLRenderPipelineDescriptor>),
    Mesh(Retained<MTLMeshRenderPipelineDescriptor>),
}

/// The ICB-capable twin of a render pipeline, compiled the first time a
/// multi-draw recorded under the pipeline lowers to an ICB.
///
/// `executeCommandsInBuffer:` needs a pipeline state created with
/// `supportIndirectCommandBuffers`, and on some drivers that flag changes how
/// direct draws render, so the twin has to be a second pipeline state. Building
/// it eagerly would double pipeline compilation for every application on
/// Metal, most of which never multi-draw at all, so the descriptor is kept
/// instead (it retains the shader functions, which the pipeline keeps alive
/// anyway) and the state is compiled on demand. A failed compilation is
/// remembered so the multi-draw falls back to the per-draw loop without
/// retrying.
pub(super) struct IcbRenderPipeline {
    descriptor: IcbPipelineDescriptor,
    label: Option<String>,
    state: OnceCell<Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>>,
}

// SAFETY: the descriptor is never mutated after `IcbRenderPipeline::new`;
// Metal pipeline descriptors are plain objects, and `MTLDevice` pipeline
// creation is thread-safe, so reading it from any thread is sound.
#[cfg(send_sync)]
unsafe impl Send for IcbRenderPipeline {}
#[cfg(send_sync)]
unsafe impl Sync for IcbRenderPipeline {}

impl core::fmt::Debug for IcbRenderPipeline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IcbRenderPipeline")
            .field("label", &self.label)
            .field("compiled", &self.state.get().is_some())
            .finish_non_exhaustive()
    }
}

impl IcbRenderPipeline {
    /// Takes the descriptor the ordinary pipeline state was created from.
    pub(super) fn new(descriptor: IcbPipelineDescriptor, label: Option<&str>) -> Self {
        match descriptor {
            IcbPipelineDescriptor::Standard(ref inner) => {
                inner.setSupportIndirectCommandBuffers(true)
            }
            IcbPipelineDescriptor::Mesh(ref inner) => inner.setSupportIndirectCommandBuffers(true),
        }
        Self {
            descriptor,
            label: label.map(ToString::to_string),
            state: OnceCell::new(),
        }
    }

    /// The compiled twin, compiling it on the first call; `None` once that
    /// has failed.
    fn state(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Option<&Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        self.state
            .get_or_init(|| {
                let result = match self.descriptor {
                    IcbPipelineDescriptor::Standard(ref inner) => {
                        device.newRenderPipelineStateWithDescriptor_error(inner)
                    }
                    IcbPipelineDescriptor::Mesh(ref inner) => device
                        .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                            inner,
                            MTLPipelineOption::empty(),
                            None,
                        ),
                };
                match result {
                    Ok(state) => Some(state),
                    Err(error) => {
                        log::debug!(
                            "could not create ICB-capable render pipeline {:?}; \
                             multi-draws recorded with it won't use ICBs: {error:?}",
                            self.label,
                        );
                        None
                    }
                }
            })
            .as_ref()
    }
}

impl super::AdapterShared {
    /// Whether this device actually executes GPU-generated render ICBs.
    ///
    /// Apple's feature tables do not predict this (see the notes on
    /// `PrivateCapabilities::indirect_command_buffers_rendering`), so it is
    /// established by running a one-draw ICB and reading the pixel back. The
    /// probe runs once, the first time a multi-draw could use an ICB, so
    /// adapters that never multi-draw never pay for it. It decides only which
    /// lowering multi-draws get, never a feature: a failed probe leaves them
    /// on the per-draw loop.
    pub(super) fn render_icb_executes(&self) -> bool {
        let mut probe = self.render_icb_probe.lock();
        *probe.get_or_insert_with(|| {
            super::icb_probe::supports_render_icb(&self.device, self.private_caps.msl_version)
        })
    }

    /// The multi-draw support kernels, compiled on first use and cached for
    /// the adapter's lifetime. A compile failure is cached too, so a broken
    /// driver doesn't recompile the library on every multi-draw call.
    ///
    /// Devices created with `MULTI_DRAW_INDIRECT_COUNT` compile them in
    /// `Adapter::open`: count draws have no lowering without them, so a
    /// failure is reported there rather than at a draw call.
    pub(super) fn icb_command_pipelines(&self) -> Result<IcbCommandPipelines, crate::DeviceError> {
        let mut pipelines = self.icb_command_pipelines.lock();
        pipelines
            .get_or_insert_with(|| IcbCommandPipelines::new(self))
            .clone()
    }

    /// Bytes Metal allocates per command in ICBs created from `descriptor`,
    /// measured once per draw kind from a small sample allocation since no API
    /// reports it ahead of time (see [`ICB_MAX_BYTES`]). `None` if even the
    /// sample cannot be allocated.
    fn icb_bytes_per_command(
        &self,
        kind_tag: u8,
        descriptor: &MTLIndirectCommandBufferDescriptor,
    ) -> Option<u64> {
        const SAMPLE_COMMANDS: usize = 512;
        let mut cache = self.icb_bytes_per_command.lock();
        let slot = &mut cache[usize::from(kind_tag)];
        if let Some(bytes) = *slot {
            return Some(bytes);
        }
        let sample = unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    descriptor,
                    SAMPLE_COMMANDS,
                    MTLResourceOptions::StorageModePrivate,
                )
        }?;
        // Rounding up keeps the bound conservative.
        let bytes = (sample.allocatedSize() as u64)
            .div_ceil(SAMPLE_COMMANDS as u64)
            .max(1);
        *slot = Some(bytes);
        Some(bytes)
    }
}

impl super::CommandEncoder {
    fn supports_icb_multi_draw(&self) -> bool {
        self.shared.private_caps.indirect_command_buffers_rendering
            && self.shared.render_icb_executes()
    }

    fn get_icb_command_pipelines(&self) -> Result<IcbCommandPipelines, crate::DeviceError> {
        self.shared.icb_command_pipelines()
    }

    fn get_icb_mesh_command_pipeline(&self) -> Option<IcbCommandPipeline> {
        self.get_icb_command_pipelines().ok()?.mesh
    }

    fn icb_primitive_type_value(
        raw_primitive_type: MTLPrimitiveType,
    ) -> Result<u32, crate::DeviceError> {
        match raw_primitive_type {
            MTLPrimitiveType::Point => Ok(ICB_PRIMITIVE_POINT),
            MTLPrimitiveType::Line => Ok(ICB_PRIMITIVE_LINE),
            MTLPrimitiveType::LineStrip => Ok(ICB_PRIMITIVE_LINE_STRIP),
            MTLPrimitiveType::Triangle => Ok(ICB_PRIMITIVE_TRIANGLE),
            MTLPrimitiveType::TriangleStrip => Ok(ICB_PRIMITIVE_TRIANGLE_STRIP),
            _ => Err(crate::DeviceError::Unexpected),
        }
    }

    /// Dispatch size for a generation kernel: one thread per draw, in
    /// threadgroups of `threadExecutionWidth` threads, rounded up so every
    /// draw is covered. The kernels ignore threads past `draw_count`.
    fn icb_generation_threadgroups(
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        draw_count: u32,
    ) -> (MTLSize, MTLSize) {
        let threads_per_threadgroup = pipeline.threadExecutionWidth().max(1);
        let threadgroup_count = (draw_count as usize).div_ceil(threads_per_threadgroup);
        (
            MTLSize {
                width: threadgroup_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_threadgroup,
                height: 1,
                depth: 1,
            },
        )
    }

    /// Try to lower a fixed-count multi-draw to a Metal indirect command
    /// buffer.
    ///
    /// This allocates the ICB and records its execution into the current
    /// render encoder immediately; the compute work that fills the ICB from
    /// `args_buffer` is queued on [`super::CommandEncoder::deferred_multi_draws`] and encoded by
    /// [`encode_deferred_multi_draws`](crate::CommandEncoder::encode_deferred_multi_draws)
    /// into the command buffer wgpu-core schedules *before* this pass, so the
    /// render pass is never split. Everything fallible happens up front:
    /// returns `false` with nothing recorded when the ICB path is unavailable,
    /// leaving the caller to record the per-draw indirect loop instead.
    /// Everything fallible about lowering `draw_count` commands to an ICB:
    /// compile the bound pipeline's ICB twin if needed, resolve the generation
    /// pipeline, bound the ICB's memory, take an ICB from the pool or create
    /// one, and encode its argument buffer. Returns the request and the
    /// pipeline state to execute it under; `None` means the caller must
    /// record a per-draw loop instead, and nothing has been recorded.
    unsafe fn prepare_icb_request(
        &mut self,
        kind: IcbDrawKind,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) -> Option<(
        IcbGenerationRequest,
        Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    )> {
        if !self.supports_icb_multi_draw() {
            return None;
        }
        // The bound pipeline's ICB-capable twin is compiled on first use.
        let icb_pipeline = self.state.render_pipeline_icb.clone()?;
        let icb_pipeline_state = icb_pipeline.state(&self.shared.device)?.clone();

        // Resolve the generation pipeline now: once execution is recorded
        // there is no falling back, so `encode_deferred_multi_draws` must not
        // be able to fail. The adapter-level cache is never cleared, which
        // makes the later lookups infallible.
        let (argument_encoder, label) = match kind {
            IcbDrawKind::Draw => {
                let Ok(pipelines) = self.get_icb_command_pipelines() else {
                    return None;
                };
                (
                    self.temp.icb_argument_encoders.draw(&pipelines.draw),
                    "wgpu multi_draw_indirect ICB",
                )
            }
            IcbDrawKind::DrawIndexed { raw_index_type, .. } => {
                let Ok(pipelines) = self.get_icb_command_pipelines() else {
                    return None;
                };
                let cache = &mut self.temp.icb_argument_encoders;
                let state = match raw_index_type {
                    MTLIndexType::UInt16 => cache.indexed_u16(&pipelines.indexed_u16),
                    MTLIndexType::UInt32 => cache.indexed_u32(&pipelines.indexed_u32),
                    _ => return None,
                };
                (state, "wgpu multi_draw_indexed_indirect ICB")
            }
            IcbDrawKind::DrawMeshTasks { .. } => {
                let pipeline = self.get_icb_mesh_command_pipeline()?;
                (
                    self.temp.icb_argument_encoders.mesh(&pipeline),
                    "wgpu multi_draw_mesh_tasks_indirect ICB",
                )
            }
        };

        let primitive_type_value = match kind {
            IcbDrawKind::DrawMeshTasks { .. } => 0,
            _ => match Self::icb_primitive_type_value(self.state.raw_primitive_type) {
                Ok(value) => value,
                Err(_) => return None,
            },
        };

        // Bound the ICB's memory before touching the pool or the allocator:
        // Metal sizes every command for the full inherited state, so a large
        // draw count is a large allocation on every GPU (see `ICB_MAX_BYTES`).
        let kind_tag = kind.tag();
        let descriptor = Self::icb_descriptor(&kind);
        let bytes_per_command = self.shared.icb_bytes_per_command(kind_tag, &descriptor)?;
        let max_commands = u32::try_from(ICB_MAX_BYTES / bytes_per_command).unwrap_or(u32::MAX);
        if draw_count > max_commands {
            return None;
        }

        // Prefer a pooled ICB of the same kind with enough capacity, taking the
        // smallest that fits so large ones stay available for large draws.
        let pooled = {
            let mut pool = self.shared.icb_pool.lock();
            let mut best: Option<usize> = None;
            for (index, entry) in pool.iter().enumerate() {
                if entry.kind_tag == kind_tag
                    && entry.capacity >= draw_count
                    && best.is_none_or(|b| pool[b].capacity > entry.capacity)
                {
                    best = Some(index);
                }
            }
            best.map(|index| pool.swap_remove(index))
        };
        let (icb, argument_buffer, capacity, bytes) = match pooled {
            Some(entry) => (
                entry.icb,
                entry.argument_buffer,
                entry.capacity,
                entry.bytes,
            ),
            None => {
                // Power-of-two capacities keep the pool reusable across
                // nearby draw counts, within the memory bound; only the first
                // `draw_count` commands are ever generated or executed.
                let capacity = draw_count
                    .checked_next_power_of_two()
                    .unwrap_or(u32::MAX)
                    .min(max_commands);
                let icb = unsafe {
                    self.shared
                        .device
                        .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                            &descriptor,
                            capacity as usize,
                            MTLResourceOptions::StorageModePrivate,
                        )
                }?;
                // Label the ICB so GPU captures and profilers attribute the
                // executed draws to wgpu's multi-draw lowering rather than an
                // anonymous ICB.
                icb.setLabel(self.shared.hal_label(label).as_deref());

                // Encode the ICB handle into an argument buffer for the
                // generation kernel; it stays valid for the ICB's whole life.
                let argument_buffer = self.shared.device.newBufferWithLength_options(
                    argument_encoder.encoded_length,
                    MTLResourceOptions::StorageModeShared,
                )?;
                argument_buffer.setLabel(
                    self.shared
                        .hal_label("wgpu ICB generation arguments")
                        .as_deref(),
                );
                unsafe {
                    argument_encoder
                        .encoder
                        .setArgumentBuffer_offset(Some(&argument_buffer), 0);
                    argument_encoder
                        .encoder
                        .setIndirectCommandBuffer_atIndex(Some(&icb), 0);
                }
                let bytes = icb.allocatedSize() as u64 + argument_buffer.allocatedSize() as u64;
                (icb, argument_buffer, capacity, bytes)
            }
        };

        let request = IcbGenerationRequest {
            kind_tag,
            capacity,
            bytes,
            kind,
            icb,
            argument_buffer,
            args_buffer: buffer.raw.clone(),
            args_offset: offset,
            draw_count,
            primitive_type_value,
        };
        Some((request, icb_pipeline_state))
    }

    /// Record the `useResource` calls every ICB execution path needs.
    fn use_icb_resources(
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        request: &IcbGenerationRequest,
    ) {
        // The ICB itself is deliberately declared with the stage-less
        // `useResource:usage:`: it is consumed by command ingestion rather
        // than by a shader stage, and declaring it stage-scoped has been
        // observed to break execution on Apple silicon (draws silently do
        // nothing). Ordering against the generation write is established by
        // `executeCommandsInBuffer` referencing the ICB directly.
        #[expect(deprecated)]
        encoder.useResource_usage(
            ProtocolObject::from_ref(&*request.icb),
            MTLResourceUsage::Read,
        );
        if let IcbDrawKind::DrawIndexed {
            ref index_buffer, ..
        } = request.kind
        {
            // The generated commands reference the index buffer via a
            // device pointer baked in at generation time, which residency
            // tracking can't see. Unlike the ICB, this is an ordinary
            // vertex-stage read, so the stage-scoped variant is correct.
            encoder.useResource_usage_stages(
                ProtocolObject::from_ref(&**index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
        }
    }

    /// Try to lower a fixed-count multi-draw to a Metal indirect command
    /// buffer.
    ///
    /// This records the ICB's execution into the current render encoder
    /// immediately; the compute work that fills the ICB from `args_buffer` is
    /// queued on [`super::CommandEncoder::deferred_multi_draws`] and encoded
    /// by [`encode_deferred_multi_draws`](crate::CommandEncoder::encode_deferred_multi_draws)
    /// into the command buffer wgpu-core schedules *before* this pass, so the
    /// render pass is never split. Everything fallible happens up front:
    /// returns `false` with nothing recorded when the ICB path is unavailable,
    /// leaving the caller to record the per-draw indirect loop instead.
    pub(super) unsafe fn defer_multi_draw_via_icb(
        &mut self,
        kind: IcbDrawKind,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) -> bool {
        if draw_count < ICB_MIN_DRAW_COUNT {
            return false;
        }
        let Some((request, icb_pipeline_state)) =
            (unsafe { self.prepare_icb_request(kind, buffer, offset, draw_count) })
        else {
            return false;
        };

        // Record execution into the render pass now; the ICB contents become
        // defined when the deferred generation runs, in a command buffer the
        // queue executes before this one.
        let pipeline = self.state.render_pipeline.as_ref().unwrap();
        let encoder = self.state.render.as_ref().unwrap();
        encoder.setRenderPipelineState(&icb_pipeline_state);
        Self::use_icb_resources(encoder, &request);
        unsafe {
            encoder.executeCommandsInBuffer_withRange(
                &request.icb,
                NSRange {
                    location: 0,
                    length: draw_count as usize,
                },
            );
        }
        encoder.setRenderPipelineState(pipeline);

        self.deferred_multi_draws
            .push(DeferredMultiDraw::Icb(request));
        true
    }

    /// Try to lower a `multi_draw_*_indirect_count` to a Metal indirect
    /// command buffer executed with a GPU-computed range.
    ///
    /// Like [`Self::defer_multi_draw_via_icb`], but the ICB holds `max_count`
    /// commands and is executed through an execution-range buffer that a
    /// deferred kernel fills with `min(count, max_count)`, keeping the draw
    /// count entirely on the GPU. Commands past the real count are generated
    /// but never executed, so no reset is needed. No draw-count threshold
    /// applies: there is no cheaper correct lowering for count draws.
    pub(super) unsafe fn defer_count_multi_draw_via_icb(
        &mut self,
        kind: IcbDrawKind,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
    ) -> bool {
        let Some((request, icb_pipeline_state)) =
            (unsafe { self.prepare_icb_request(kind, buffer, offset, max_count) })
        else {
            return false;
        };
        let Some(range_buffer) = self.shared.device.newBufferWithLength_options(
            size_of::<MTLIndirectCommandBufferExecutionRange>(),
            MTLResourceOptions::StorageModePrivate,
        ) else {
            return false;
        };
        range_buffer.setLabel(self.shared.hal_label("wgpu ICB execution range").as_deref());

        let pipeline = self.state.render_pipeline.as_ref().unwrap();
        let encoder = self.state.render.as_ref().unwrap();
        encoder.setRenderPipelineState(&icb_pipeline_state);
        Self::use_icb_resources(encoder, &request);
        #[expect(deprecated)]
        encoder.useResource_usage(
            ProtocolObject::from_ref(&*range_buffer),
            MTLResourceUsage::Read,
        );
        unsafe {
            encoder.executeCommandsInBuffer_indirectBuffer_indirectBufferOffset(
                &request.icb,
                &range_buffer,
                0,
            );
        }
        encoder.setRenderPipelineState(pipeline);

        self.deferred_multi_draws.push(DeferredMultiDraw::IcbCount {
            request,
            count_buffer: count_buffer.raw.clone(),
            count_offset,
            range_buffer,
        });
        true
    }

    /// Count-draw fallback for pipelines that can't execute inside an ICB and
    /// for counts past the ICB memory bound: queue a kernel that copies
    /// `min(count, max_count)` argument structs from `buffer` into a private
    /// buffer and zeroes the rest, so a fixed `max_count`-length per-draw loop
    /// over the result is correct. Returns that buffer, or `None` (with the
    /// reason logged) when the draw cannot be recorded at all; nothing has
    /// been queued in that case, and the caller records no draws.
    pub(super) unsafe fn defer_clamped_count_args(
        &mut self,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        count_buffer: &super::Buffer,
        count_offset: wgt::BufferAddress,
        max_count: u32,
        words_per_draw: u32,
    ) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
        // The kernels were compiled when the device was created with
        // `MULTI_DRAW_INDIRECT_COUNT`, so this only fails on a driver that has
        // since broken; there is no lowering without them.
        if let Err(error) = self.get_icb_command_pipelines() {
            log::error!(
                "Metal multi-draw support kernels are unavailable ({error}); \
                 a multi_draw_*_indirect_count call is not recorded"
            );
            return None;
        }
        let length = max_count as usize * words_per_draw as usize * WORD_SIZE;
        let Some(dst_buffer) = self
            .shared
            .device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModePrivate)
        else {
            log::error!(
                "Metal could not allocate {length} bytes for clamped multi-draw \
                 arguments; a multi_draw_*_indirect_count call is not recorded"
            );
            return None;
        };
        dst_buffer.setLabel(
            self.shared
                .hal_label("wgpu clamped multi-draw args")
                .as_deref(),
        );

        self.deferred_multi_draws
            .push(DeferredMultiDraw::ClampedArgs {
                dst_buffer: dst_buffer.clone(),
                src_buffer: buffer.raw.clone(),
                src_offset: offset,
                count_buffer: count_buffer.raw.clone(),
                count_offset,
                max_count,
                words_per_draw,
            });
        Some(dst_buffer)
    }

    /// The descriptor for ICBs that execute `kind` draws. ICBs are only reused
    /// across draws whose descriptors match, which [`IcbDrawKind::tag`] tracks.
    fn icb_descriptor(kind: &IcbDrawKind) -> Retained<MTLIndirectCommandBufferDescriptor> {
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setInheritPipelineState(true);
        descriptor.setInheritBuffers(true);
        match kind {
            IcbDrawKind::Draw => {
                descriptor.setCommandTypes(MTLIndirectCommandType::Draw);
                descriptor.setMaxVertexBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
                descriptor.setMaxFragmentBufferBindCount(0);
            }
            IcbDrawKind::DrawIndexed { .. } => {
                descriptor.setCommandTypes(MTLIndirectCommandType::DrawIndexed);
                descriptor.setMaxVertexBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
                descriptor.setMaxFragmentBufferBindCount(0);
            }
            IcbDrawKind::DrawMeshTasks { .. } => {
                descriptor.setCommandTypes(MTLIndirectCommandType::DrawMeshThreadgroups);
                descriptor.setMaxFragmentBufferBindCount(0);
                unsafe {
                    descriptor.setMaxObjectBufferBindCount(0);
                    descriptor.setMaxMeshBufferBindCount(0);
                }
            }
        }
        descriptor
    }

    /// Encode the generation (and, where it pays, optimize) work for every
    /// multi-draw queued since the last call; see the module documentation.
    pub(super) unsafe fn encode_deferred_icb_generation(&mut self) {
        if self.deferred_multi_draws.is_empty() {
            return;
        }
        // wgpu-core records this right after the pass's texture-init fix-ups,
        // which may have left a blit encoder open; Metal aborts the process if
        // a second encoder is created while one is encoding.
        self.leave_blit();

        // The pipelines were resolved when each request was queued and the
        // adapter-level cache is never cleared, so these lookups cannot fail.
        let pipelines = self.get_icb_command_pipelines().unwrap();
        let mesh_pipeline = self
            .deferred_multi_draws
            .iter()
            .any(|draw| {
                matches!(
                    draw.icb_request(),
                    Some(IcbGenerationRequest {
                        kind: IcbDrawKind::DrawMeshTasks { .. },
                        ..
                    })
                )
            })
            .then(|| self.get_icb_mesh_command_pipeline().unwrap());

        // No reset pass: the generation kernels write every command in the
        // executed range, calling `reset()` on the slots whose draw is empty,
        // and count draws execute a GPU-clamped range that excludes the rest.

        // A single labeled compute encoder holds every generation dispatch
        // for the pass, which keeps encoder switches minimal and gives GPU
        // captures/profilers one legible "ICB generation" node per pass.
        let raw = self.raw_cmd_buf.as_ref().unwrap();
        let compute = raw.computeCommandEncoder().unwrap();
        compute.setLabel(
            self.shared
                .hal_label("wgpu multi-draw ICB generation")
                .as_deref(),
        );
        for draw in &self.deferred_multi_draws {
            match draw {
                DeferredMultiDraw::Icb(request) | DeferredMultiDraw::IcbCount { request, .. } => {
                    let pipeline = match request.kind {
                        IcbDrawKind::Draw => &pipelines.draw,
                        IcbDrawKind::DrawIndexed { raw_index_type, .. } => {
                            if raw_index_type == MTLIndexType::UInt16 {
                                &pipelines.indexed_u16
                            } else {
                                &pipelines.indexed_u32
                            }
                        }
                        IcbDrawKind::DrawMeshTasks { .. } => mesh_pipeline.as_ref().unwrap(),
                    };
                    compute.setComputePipelineState(&pipeline.pipeline);
                    unsafe {
                        compute.setBuffer_offset_atIndex(Some(&request.argument_buffer), 0, 0);
                        compute.setBuffer_offset_atIndex(
                            Some(&request.args_buffer),
                            request.args_offset as usize,
                            1,
                        );
                        match request.kind {
                            IcbDrawKind::Draw => {
                                compute.setBytes_length_atIndex(
                                    NonNull::from(&request.primitive_type_value).cast(),
                                    size_of::<u32>(),
                                    2,
                                );
                                compute.setBytes_length_atIndex(
                                    NonNull::from(&request.draw_count).cast(),
                                    size_of::<u32>(),
                                    3,
                                );
                            }
                            IcbDrawKind::DrawIndexed {
                                ref index_buffer,
                                index_offset,
                                ..
                            } => {
                                compute.setBuffer_offset_atIndex(
                                    Some(index_buffer),
                                    index_offset as usize,
                                    2,
                                );
                                compute.setBytes_length_atIndex(
                                    NonNull::from(&request.primitive_type_value).cast(),
                                    size_of::<u32>(),
                                    3,
                                );
                                compute.setBytes_length_atIndex(
                                    NonNull::from(&request.draw_count).cast(),
                                    size_of::<u32>(),
                                    4,
                                );
                            }
                            IcbDrawKind::DrawMeshTasks {
                                ref threadgroup_sizes,
                            } => {
                                compute.setBytes_length_atIndex(
                                    NonNull::new(threadgroup_sizes.as_ptr().cast_mut().cast())
                                        .unwrap(),
                                    size_of::<[u32; 6]>(),
                                    2,
                                );
                                compute.setBytes_length_atIndex(
                                    NonNull::from(&request.draw_count).cast(),
                                    size_of::<u32>(),
                                    3,
                                );
                            }
                        }
                        compute.useResource_usage(
                            ProtocolObject::from_ref(&*request.icb),
                            MTLResourceUsage::Write,
                        );
                    }
                    let (threadgroups, threads_per_threadgroup) =
                        Self::icb_generation_threadgroups(&pipeline.pipeline, request.draw_count);
                    compute.dispatchThreadgroups_threadsPerThreadgroup(
                        threadgroups,
                        threads_per_threadgroup,
                    );

                    if let DeferredMultiDraw::IcbCount {
                        count_buffer,
                        count_offset,
                        range_buffer,
                        ..
                    } = draw
                    {
                        // Clamp the GPU-resident count into the execution
                        // range this ICB executes with.
                        compute.setComputePipelineState(&pipelines.execution_range.pipeline);
                        unsafe {
                            compute.setBuffer_offset_atIndex(
                                Some(count_buffer),
                                *count_offset as usize,
                                0,
                            );
                            compute.setBuffer_offset_atIndex(Some(range_buffer), 0, 1);
                            compute.setBytes_length_atIndex(
                                NonNull::from(&request.draw_count).cast(),
                                size_of::<u32>(),
                                2,
                            );
                        }
                        let single = MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        };
                        compute.dispatchThreadgroups_threadsPerThreadgroup(single, single);
                    }
                }
                DeferredMultiDraw::ClampedArgs {
                    dst_buffer,
                    src_buffer,
                    src_offset,
                    count_buffer,
                    count_offset,
                    max_count,
                    words_per_draw,
                } => {
                    compute.setComputePipelineState(&pipelines.clamp.pipeline);
                    unsafe {
                        compute.setBuffer_offset_atIndex(Some(dst_buffer), 0, 0);
                        compute.setBuffer_offset_atIndex(Some(src_buffer), *src_offset as usize, 1);
                        compute.setBuffer_offset_atIndex(
                            Some(count_buffer),
                            *count_offset as usize,
                            2,
                        );
                        compute.setBytes_length_atIndex(
                            NonNull::from(max_count).cast(),
                            size_of::<u32>(),
                            3,
                        );
                        compute.setBytes_length_atIndex(
                            NonNull::from(words_per_draw).cast(),
                            size_of::<u32>(),
                            4,
                        );
                    }
                    let (threadgroups, threads_per_threadgroup) = Self::icb_generation_threadgroups(
                        &pipelines.clamp.pipeline,
                        max_count * words_per_draw,
                    );
                    compute.dispatchThreadgroups_threadsPerThreadgroup(
                        threadgroups,
                        threads_per_threadgroup,
                    );
                }
            }
        }
        compute.endEncoding();

        // Let Metal strip inherited state the generated commands don't need.
        // This halves ICB execution time on Apple3 and costs 5-10% on every
        // later Apple GPU measured, so it runs only where it pays.
        if self.shared.private_caps.indirect_command_buffers_optimize
            && self
                .deferred_multi_draws
                .iter()
                .any(|draw| draw.icb_request().is_some())
        {
            let group = self.shared.hal_label("wgpu optimize multi-draw ICBs");
            let blit = self.enter_blit();
            if let Some(ref group) = group {
                blit.pushDebugGroup(group);
            }
            for draw in &self.deferred_multi_draws {
                let Some(request) = draw.icb_request() else {
                    continue;
                };
                unsafe {
                    blit.optimizeIndirectCommandBuffer_withRange(
                        &request.icb,
                        NSRange {
                            location: 0,
                            length: request.draw_count as usize,
                        },
                    );
                }
            }
            if group.is_some() {
                blit.popDebugGroup();
            }
        }
        self.leave_blit();

        // Drain rather than take so the request vector keeps its allocation
        // for the next pass.
        let shared = self.shared.clone();
        self.deferred_multi_draw_resources
            .extend(self.deferred_multi_draws.drain(..).map(|draw| match draw {
                DeferredMultiDraw::Icb(request) => IcbExecutionResources {
                    shared: shared.clone(),
                    kind_tag: request.kind_tag,
                    capacity: request.capacity,
                    bytes: request.bytes,
                    icb: Some(request.icb),
                    argument_buffer: Some(request.argument_buffer),
                    _extra: Vec::new(),
                },
                DeferredMultiDraw::IcbCount {
                    request,
                    range_buffer,
                    ..
                } => IcbExecutionResources {
                    shared: shared.clone(),
                    kind_tag: request.kind_tag,
                    capacity: request.capacity,
                    bytes: request.bytes,
                    icb: Some(request.icb),
                    argument_buffer: Some(request.argument_buffer),
                    _extra: alloc::vec![range_buffer],
                },
                DeferredMultiDraw::ClampedArgs { dst_buffer, .. } => IcbExecutionResources {
                    shared: shared.clone(),
                    kind_tag: 0,
                    capacity: 0,
                    bytes: 0,
                    icb: None,
                    argument_buffer: None,
                    _extra: alloc::vec![dst_buffer],
                },
            }));
    }
}
