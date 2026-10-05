//! Lowering of fixed-count multi-draws to Metal indirect command buffers.
//!
//! A multi-draw records the execution of an ICB into the current render
//! pass immediately; the compute work that fills the ICB from the indirect
//! argument buffer is deferred to [`encode_deferred_icb_generation`], which
//! wgpu-core schedules ahead of the pass so the pass is never split.
//!
//! [`encode_deferred_icb_generation`]: super::CommandEncoder::encode_deferred_icb_generation

use alloc::{sync::Arc, vec::Vec};
use core::ptr::NonNull;
use wgpu_sync::Mutex;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLArgumentEncoder, MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFunction, MTLIndexType,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLLibrary, MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderStages, MTLResource,
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
///
/// Draw ICBs keep a fragment count of 0: the same A12 draws correctly with a
/// fragment-stage buffer at slot 0 or 5 and a count of 0, as do the A14, A18
/// Pro and M4 Max, while a count of 31 grows every A12 command from 1610 to
/// 2114 bytes. Mesh ICBs only exist on Apple9, where the full table costs
/// nothing, so they use it for every stage.
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

/// [`IcbDrawKind::tag`] values. Each kind has its own ICB descriptor, and ICBs
/// are only reused for the same kind.
pub(super) const ICB_KIND_DRAW: u8 = 0;
const ICB_KIND_DRAW_INDEXED: u8 = 1;
const ICB_KIND_MESH: u8 = 2;

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
/// commands, compiled once per device by [`IcbContext::new`].
#[derive(Clone, Debug)]
pub(super) struct IcbCommandPipelines {
    draw: IcbCommandPipeline,
    indexed_u16: IcbCommandPipeline,
    indexed_u32: IcbCommandPipeline,
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

impl IcbDrawKind {
    /// Which ICB descriptor (command type and bind counts) a draw kind needs;
    /// ICBs are only ever reused for the same tag.
    fn tag(&self) -> u8 {
        match self {
            IcbDrawKind::Draw => ICB_KIND_DRAW,
            IcbDrawKind::DrawIndexed { .. } => ICB_KIND_DRAW_INDEXED,
            IcbDrawKind::DrawMeshTasks { .. } => ICB_KIND_MESH,
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
/// [`super::CommandBuffer::_icb_resources`]. Dropped with that command buffer.
/// The `encode_deferred_multi_draws` contract makes that no earlier than the
/// completion of the pass that executes the ICB, which is when the ICB can
/// safely return to the pool.
pub(super) struct IcbExecutionResources {
    context: Arc<IcbContext>,
    kind_tag: u8,
    capacity: u32,
    bytes: u64,
    icb: Option<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    argument_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
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
        let mut pool = self.context.pool.lock();
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

/// Per-device state of the ICB lowering. [`IcbContext::new`] sets it up when
/// the device is opened, so recording a multi-draw never compiles, probes or
/// measures anything.
pub(super) struct IcbContext {
    pipelines: IcbCommandPipelines,
    /// Bytes Metal allocates per ICB command, indexed by [`IcbDrawKind::tag`];
    /// 0 for a kind this device does not lower.
    bytes_per_command: [u64; 3],
    /// Indirect command buffers awaiting reuse; see [`PooledIcb`].
    pool: Mutex<Vec<PooledIcb>>,
}

impl core::fmt::Debug for IcbContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IcbContext")
            .field("bytes_per_command", &self.bytes_per_command)
            .finish_non_exhaustive()
    }
}

impl IcbContext {
    /// Sets up the ICB lowering for a device opened on `shared`, or returns
    /// `None` when multi-draws keep the per-draw loop: the lowering is off for
    /// this adapter (always under `STRICT_WEBGPU_COMPLIANCE`), the device
    /// fails the execution probe, or a generation kernel fails to compile.
    pub(super) fn new(shared: &super::AdapterShared) -> Option<Self> {
        let caps = &shared.private_caps;
        if !caps.indirect_command_buffers_rendering {
            return None;
        }
        // Apple's feature tables don't predict whether a device executes
        // GPU-generated render ICBs (see the notes in `adapter.rs`), so run
        // one and read the pixel back.
        if !super::icb_probe::supports_render_icb(
            &shared.device,
            caps.msl_version,
            caps.indirect_command_buffers_optimize,
        ) {
            log::debug!(
                "Metal did not execute the render ICB probe correctly, so multi-draws \
                 will use the per-draw loop"
            );
            return None;
        }
        let mut pipelines = IcbCommandPipelines::new(shared).ok()?;
        let mut bytes_per_command = [0; 3];
        for kind_tag in [ICB_KIND_DRAW, ICB_KIND_DRAW_INDEXED] {
            bytes_per_command[usize::from(kind_tag)] =
                sample_bytes_per_command(&shared.device, kind_tag)?;
        }
        if pipelines.mesh.is_some() {
            match sample_bytes_per_command(&shared.device, ICB_KIND_MESH) {
                Some(bytes) => bytes_per_command[usize::from(ICB_KIND_MESH)] = bytes,
                None => pipelines.mesh = None,
            }
        }
        log::debug!(
            "Metal will run multi-draws of {ICB_MIN_DRAW_COUNT} or more draws through \
             indirect command buffers (mesh tasks: {})",
            pipelines.mesh.is_some(),
        );
        Some(Self {
            pipelines,
            bytes_per_command,
            pool: Mutex::new(Vec::new()),
        })
    }

    /// Whether mesh-task multi-draws lower to ICBs on this device.
    pub(super) fn lowers_mesh_draws(&self) -> bool {
        self.pipelines.mesh.is_some()
    }
}

/// Bytes Metal allocates per command in ICBs of `kind_tag`, measured from a
/// small sample allocation since no API reports it ahead of time (see
/// [`ICB_MAX_BYTES`]). `None` if even the sample cannot be allocated.
fn sample_bytes_per_command(device: &ProtocolObject<dyn MTLDevice>, kind_tag: u8) -> Option<u64> {
    const SAMPLE_COMMANDS: usize = 512;
    let sample = unsafe {
        device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
            &icb_descriptor(kind_tag),
            SAMPLE_COMMANDS,
            MTLResourceOptions::StorageModePrivate,
        )
    }?;
    // Rounding up keeps the bound conservative.
    Some(
        (sample.allocatedSize() as u64)
            .div_ceil(SAMPLE_COMMANDS as u64)
            .max(1),
    )
}

/// The descriptor for ICBs of `kind_tag`. The execution probe uses the same
/// one, so a fault that depends on the descriptor fails the probe (and leaves
/// multi-draws on the per-draw loop) instead of a draw.
pub(super) fn icb_descriptor(kind_tag: u8) -> Retained<MTLIndirectCommandBufferDescriptor> {
    let descriptor = MTLIndirectCommandBufferDescriptor::new();
    descriptor.setInheritPipelineState(true);
    descriptor.setInheritBuffers(true);
    match kind_tag {
        ICB_KIND_DRAW => {
            descriptor.setCommandTypes(MTLIndirectCommandType::Draw);
            descriptor.setMaxVertexBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
            descriptor.setMaxFragmentBufferBindCount(0);
        }
        ICB_KIND_DRAW_INDEXED => {
            descriptor.setCommandTypes(MTLIndirectCommandType::DrawIndexed);
            descriptor.setMaxVertexBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
            descriptor.setMaxFragmentBufferBindCount(0);
        }
        _ => {
            descriptor.setCommandTypes(MTLIndirectCommandType::DrawMeshThreadgroups);
            descriptor.setMaxFragmentBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
            unsafe {
                descriptor.setMaxObjectBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
                descriptor.setMaxMeshBufferBindCount(ICB_MAX_INHERITED_BUFFER_BIND_COUNT);
            }
        }
    }
    descriptor
}

impl super::CommandEncoder {
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
        // Both are fixed when the device and the pipeline are created; see
        // `IcbContext::new` and `create_render_pipeline`.
        let Some(context) = self.icb.clone() else {
            return false;
        };
        let Some(icb_pipeline_state) = self.state.render_pipeline_icb.clone() else {
            return false;
        };

        let (argument_encoder, label) = match kind {
            IcbDrawKind::Draw => (
                self.temp
                    .icb_argument_encoders
                    .draw(&context.pipelines.draw),
                "wgpu multi_draw_indirect ICB",
            ),
            IcbDrawKind::DrawIndexed { raw_index_type, .. } => {
                let cache = &mut self.temp.icb_argument_encoders;
                let state = match raw_index_type {
                    MTLIndexType::UInt16 => cache.indexed_u16(&context.pipelines.indexed_u16),
                    MTLIndexType::UInt32 => cache.indexed_u32(&context.pipelines.indexed_u32),
                    _ => return false,
                };
                (state, "wgpu multi_draw_indexed_indirect ICB")
            }
            IcbDrawKind::DrawMeshTasks { .. } => {
                let Some(ref pipeline) = context.pipelines.mesh else {
                    return false;
                };
                (
                    self.temp.icb_argument_encoders.mesh(pipeline),
                    "wgpu multi_draw_mesh_tasks_indirect ICB",
                )
            }
        };

        let primitive_type_value = match kind {
            IcbDrawKind::DrawMeshTasks { .. } => 0,
            _ => match Self::icb_primitive_type_value(self.state.raw_primitive_type) {
                Ok(value) => value,
                Err(_) => return false,
            },
        };

        // Bound the ICB's memory before touching the pool or the allocator:
        // Metal sizes every command for the full inherited state, so a large
        // draw count is a large allocation on every GPU (see `ICB_MAX_BYTES`).
        let kind_tag = kind.tag();
        let bytes_per_command = context.bytes_per_command[usize::from(kind_tag)].max(1);
        let max_commands = u32::try_from(ICB_MAX_BYTES / bytes_per_command).unwrap_or(u32::MAX);
        if draw_count > max_commands {
            return false;
        }

        // Prefer a pooled ICB of the same kind with enough capacity, taking the
        // smallest that fits so large ones stay available for large draws.
        let pooled = {
            let mut pool = context.pool.lock();
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
                let Some(icb) = (unsafe {
                    self.shared
                        .device
                        .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                            &icb_descriptor(kind_tag),
                            capacity as usize,
                            MTLResourceOptions::StorageModePrivate,
                        )
                }) else {
                    return false;
                };
                // Label the ICB so GPU captures and profilers attribute the
                // executed draws to wgpu's multi-draw lowering rather than an
                // anonymous ICB.
                icb.setLabel(self.shared.hal_label(label).as_deref());

                // Encode the ICB handle into an argument buffer for the
                // generation kernel; it stays valid for the ICB's whole life.
                let Some(argument_buffer) = self.shared.device.newBufferWithLength_options(
                    argument_encoder.encoded_length,
                    MTLResourceOptions::StorageModeShared,
                ) else {
                    return false;
                };
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

        // Record execution into the render pass now; the ICB contents become
        // defined when the deferred generation runs, in a command buffer the
        // queue executes before this one.
        let pipeline = self.state.render_pipeline.as_ref().unwrap();
        let encoder = self.state.render.as_ref().unwrap();
        encoder.setRenderPipelineState(&icb_pipeline_state);
        #[expect(deprecated)]
        unsafe {
            encoder.useResource_usage(ProtocolObject::from_ref(&*icb), MTLResourceUsage::Read);
            if let IcbDrawKind::DrawIndexed {
                ref index_buffer, ..
            } = kind
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
            encoder.executeCommandsInBuffer_withRange(
                &icb,
                NSRange {
                    location: 0,
                    length: draw_count as usize,
                },
            );
        }
        encoder.setRenderPipelineState(pipeline);

        self.deferred_multi_draws.push(IcbGenerationRequest {
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
        });
        true
    }

    /// Encode the generate (and, on Apple3, optimize) work for every
    /// multi-draw queued since the last call; see the module documentation.
    pub(super) unsafe fn encode_deferred_icb_generation(&mut self) {
        if self.deferred_multi_draws.is_empty() {
            return;
        }
        // Requests are only queued when the encoder has a context.
        let Some(context) = self.icb.clone() else {
            return;
        };
        // wgpu-core records this right after the pass's texture-init fix-ups,
        // which may have left a blit encoder open, and the hal contract allows
        // an open acceleration-structure encoder here too. Metal aborts the
        // process if a second encoder is created while one is encoding.
        self.leave_acceleration_structure_builder();
        self.leave_blit();
        let pipelines = &context.pipelines;

        // No reset pass: the generation kernels write every command in the
        // executed range, calling `reset()` on the slots whose draw is empty,
        // so a reset blit would only repeat that work.

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
        for request in &self.deferred_multi_draws {
            let pipeline = match request.kind {
                IcbDrawKind::Draw => &pipelines.draw,
                IcbDrawKind::DrawIndexed { raw_index_type, .. } => {
                    if raw_index_type == MTLIndexType::UInt16 {
                        &pipelines.indexed_u16
                    } else {
                        &pipelines.indexed_u32
                    }
                }
                // Mesh requests are only queued when the mesh kernel exists.
                IcbDrawKind::DrawMeshTasks { .. } => pipelines.mesh.as_ref().unwrap(),
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
                            NonNull::new(threadgroup_sizes.as_ptr().cast_mut().cast()).unwrap(),
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
            compute
                .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        }
        compute.endEncoding();

        // Let Metal strip inherited state the generated commands don't need.
        // This halves ICB execution time on Apple3 and costs 5-10% on every
        // later Apple GPU measured, so it runs only where it pays.
        if self.shared.private_caps.indirect_command_buffers_optimize {
            let group = self.shared.hal_label("wgpu optimize multi-draw ICBs");
            let blit = self.enter_blit();
            if let Some(ref group) = group {
                blit.pushDebugGroup(group);
            }
            for request in &self.deferred_multi_draws {
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
        self.deferred_multi_draw_resources
            .extend(
                self.deferred_multi_draws
                    .drain(..)
                    .map(|request| IcbExecutionResources {
                        context: context.clone(),
                        kind_tag: request.kind_tag,
                        capacity: request.capacity,
                        bytes: request.bytes,
                        icb: Some(request.icb),
                        argument_buffer: Some(request.argument_buffer),
                    }),
            );
    }
}
