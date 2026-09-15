//! Lowering of fixed-count multi-draws to Metal indirect command buffers.
//!
//! A multi-draw records the execution of an ICB into the current render
//! pass immediately; the compute work that fills the ICB from the indirect
//! argument buffer is deferred to [`encode_deferred_icb_generation`], which
//! wgpu-core schedules ahead of the pass so the pass is never split.
//!
//! [`encode_deferred_icb_generation`]: super::CommandEncoder::encode_deferred_icb_generation

use alloc::sync::Arc;
use core::ptr::NonNull;

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
/// inherited, and Apple3, Apple7 and Apple9 GPUs do accept 0. An A12 (Apple5,
/// iOS 18.7) does not: an ICB whose count leaves out a vertex-buffer slot the
/// inherited pipeline reads faults the GPU on execution -- no error, no
/// validation-layer assertion, Metal just stops executing every later command
/// buffer. wgpu binds vertex buffers from the top of the 31-slot argument
/// table, so only the full table covers every layout.
const ICB_MAX_INHERITED_BUFFER_BIND_COUNT: usize = 31;

/// Bounds on the per-adapter pool of indirect command buffers: entries kept,
/// and the total command capacity they may hold (256K commands).
const ICB_POOL_MAX_ENTRIES: usize = 8;
const ICB_POOL_MAX_COMMANDS: u32 = 1 << 18;

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
        super::device::compile_msl_library(&shared.device, shared.private_caps.msl_version, source)
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
    icb: Option<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    argument_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl core::fmt::Debug for IcbExecutionResources {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IcbExecutionResources")
            .field("kind_tag", &self.kind_tag)
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}

impl Drop for IcbExecutionResources {
    fn drop(&mut self) {
        let (Some(icb), Some(argument_buffer)) = (self.icb.take(), self.argument_buffer.take())
        else {
            return;
        };
        if self.capacity > ICB_POOL_MAX_COMMANDS {
            return;
        }
        let mut pool = self.shared.icb_pool.lock();
        let mut total: u32 = pool.iter().map(|entry| entry.capacity).sum();
        while pool.len() >= ICB_POOL_MAX_ENTRIES || total + self.capacity > ICB_POOL_MAX_COMMANDS {
            // Make room by evicting the smallest entry, unless this one is
            // smaller still: large ICBs are the expensive ones to recreate.
            let Some((index, smallest)) = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.capacity)
                .map(|(index, entry)| (index, entry.capacity))
            else {
                break;
            };
            if smallest >= self.capacity {
                return;
            }
            total -= smallest;
            pool.swap_remove(index);
        }
        pool.push(PooledIcb {
            kind_tag: self.kind_tag,
            capacity: self.capacity,
            icb,
            argument_buffer,
        });
    }
}

impl super::AdapterShared {
    /// Whether this device actually executes GPU-generated render ICBs.
    ///
    /// Apple's feature tables do not predict this (see the notes on
    /// `PrivateCapabilities::indirect_command_buffers_rendering`), so it is
    /// established by running a one-draw ICB and reading the pixel back. The
    /// probe runs once, the first time a multi-draw could use an ICB, so
    /// adapters that never multi-draw never pay for it.
    pub(super) fn render_icb_executes(&self) -> bool {
        let mut probe = self.render_icb_probe.lock();
        *probe.get_or_insert_with(|| {
            super::icb_probe::supports_render_icb(&self.device, self.private_caps.msl_version)
        })
    }
}

impl super::CommandEncoder {
    fn supports_icb_multi_draw(&self) -> bool {
        self.shared.private_caps.indirect_command_buffers_rendering
            && self.shared.render_icb_executes()
    }

    fn get_icb_command_pipelines(&self) -> Result<IcbCommandPipelines, crate::DeviceError> {
        let mut pipelines = self.shared.icb_command_pipelines.lock();
        // A compile failure is cached so a broken driver doesn't recompile
        // the generation library on every multi-draw call.
        pipelines
            .get_or_insert_with(|| IcbCommandPipelines::new(&self.shared))
            .clone()
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
    pub(super) unsafe fn defer_multi_draw_via_icb(
        &mut self,
        kind: IcbDrawKind,
        buffer: &super::Buffer,
        offset: wgt::BufferAddress,
        draw_count: u32,
    ) -> bool {
        if draw_count < ICB_MIN_DRAW_COUNT
            || !self.supports_icb_multi_draw()
            || self.state.render_pipeline_icb.is_none()
        {
            return false;
        }

        // Resolve the generation pipeline now: once execution is recorded
        // there is no falling back, so `encode_deferred_multi_draws` must not
        // be able to fail. The adapter-level cache is never cleared, which
        // makes the later lookups infallible.
        let (argument_encoder, label) = match kind {
            IcbDrawKind::Draw => {
                let Ok(pipelines) = self.get_icb_command_pipelines() else {
                    return false;
                };
                (
                    self.temp.icb_argument_encoders.draw(&pipelines.draw),
                    "wgpu multi_draw_indirect ICB",
                )
            }
            IcbDrawKind::DrawIndexed { raw_index_type, .. } => {
                let Ok(pipelines) = self.get_icb_command_pipelines() else {
                    return false;
                };
                let cache = &mut self.temp.icb_argument_encoders;
                let state = match raw_index_type {
                    MTLIndexType::UInt16 => cache.indexed_u16(&pipelines.indexed_u16),
                    MTLIndexType::UInt32 => cache.indexed_u32(&pipelines.indexed_u32),
                    _ => return false,
                };
                (state, "wgpu multi_draw_indexed_indirect ICB")
            }
            IcbDrawKind::DrawMeshTasks { .. } => {
                let Some(pipeline) = self.get_icb_mesh_command_pipeline() else {
                    return false;
                };
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
                Err(_) => return false,
            },
        };

        let kind_tag = kind.tag();
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
        let (icb, argument_buffer, capacity) = match pooled {
            Some(entry) => (entry.icb, entry.argument_buffer, entry.capacity),
            None => {
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
                // Power-of-two capacities keep the pool reusable across
                // nearby draw counts; only the first `draw_count` commands are
                // ever generated or executed.
                let capacity = draw_count.next_power_of_two();
                let Some(icb) = (unsafe {
                    self.shared
                        .device
                        .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                            &descriptor,
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
                (icb, argument_buffer, capacity)
            }
        };

        // Record execution into the render pass now; the ICB contents become
        // defined when the deferred generation runs, in a command buffer the
        // queue executes before this one.
        let pipeline = self.state.render_pipeline.as_ref().unwrap();
        let icb_pipeline = self.state.render_pipeline_icb.as_ref().unwrap();
        let encoder = self.state.render.as_ref().unwrap();
        encoder.setRenderPipelineState(icb_pipeline);
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

    /// Encode the reset/generate/optimize work for every multi-draw queued
    /// since the last call; see the module documentation.
    pub(super) unsafe fn encode_deferred_icb_generation(&mut self) {
        if self.deferred_multi_draws.is_empty() {
            return;
        }
        // The pipelines were resolved when each request was queued and the
        // adapter-level cache is never cleared, so these lookups cannot fail.
        let pipelines = self.get_icb_command_pipelines().unwrap();
        let mesh_pipeline = self
            .deferred_multi_draws
            .iter()
            .any(|request| matches!(request.kind, IcbDrawKind::DrawMeshTasks { .. }))
            .then(|| self.get_icb_mesh_command_pipeline().unwrap());

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
            let blit = self.enter_blit();
            blit.pushDebugGroup(&NSString::from_str("wgpu optimize multi-draw ICBs"));
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
            blit.popDebugGroup();
        }
        self.leave_blit();

        // Drain rather than take so the request vector keeps its allocation
        // for the next pass.
        self.deferred_multi_draw_resources
            .extend(
                self.deferred_multi_draws
                    .drain(..)
                    .map(|request| IcbExecutionResources {
                        shared: self.shared.clone(),
                        kind_tag: request.kind_tag,
                        capacity: request.capacity,
                        icb: Some(request.icb),
                        argument_buffer: Some(request.argument_buffer),
                    }),
            );
    }
}
