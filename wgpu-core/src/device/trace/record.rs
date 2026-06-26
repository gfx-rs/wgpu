use alloc::{borrow::Cow, string::ToString, sync::Arc, vec::Vec};
use core::{any::Any, convert::Infallible, marker::PhantomData};
use std::io::Write as _;

use crate::{
    command::{
        ArcCommand, ArcComputeCommand, ArcPassTimestampWrites, ArcReferences, ArcRenderCommand,
        BasePass, ColorAttachments, Command, ComputeCommand, PointerReferences, RenderCommand,
        RenderPassColorAttachment, ResolvedRenderPassDepthStencilAttachment,
    },
    device::trace::{Data, DataKind},
    id::{markers, PointerId},
    storage::StorageItem,
};

use super::{
    Action, TraceBindGroupDescriptor, TraceComputePipelineDescriptor,
    TraceGeneralRenderPipelineDescriptor, FILE_NAME,
};

pub(crate) fn new_render_bundle_encoder_descriptor(
    label: crate::Label<'_>,
    context: &crate::device::RenderPassContext,
    depth_read_only: bool,
    stencil_read_only: bool,
) -> crate::command::RenderBundleEncoderDescriptor<'static> {
    crate::command::RenderBundleEncoderDescriptor {
        label: label.map(|l| Cow::from(l.to_string())),
        color_formats: Cow::from(context.attachments.colors.to_vec()),
        depth_stencil: context.attachments.depth_stencil.map(|format| {
            wgt::RenderBundleDepthStencil {
                format,
                depth_read_only,
                stencil_read_only,
            }
        }),
        sample_count: context.sample_count,
        multiview: context.multiview_mask,
    }
}

pub trait Trace: Any + Send + Sync {
    fn make_binary(&mut self, kind: DataKind, data: &[u8]) -> Data;

    fn make_string(&mut self, kind: DataKind, data: &str) -> Data;

    fn add(&mut self, action: Action<'_, PointerReferences>)
    where
        for<'a> Action<'a, PointerReferences>: serde::Serialize;
}

#[derive(Debug)]
pub struct DiskTrace {
    path: std::path::PathBuf,
    file: std::fs::File,
    config: ron::ser::PrettyConfig,
    data_id: usize,
}

impl DiskTrace {
    pub fn new(path: std::path::PathBuf) -> Result<Self, std::io::Error> {
        log::debug!("Tracing into '{path:?}'");
        let mut file = std::fs::File::create(path.join(FILE_NAME))?;
        file.write_all(b"[\n")?;
        Ok(Self {
            path,
            file,
            config: ron::ser::PrettyConfig::default(),
            data_id: 0,
        })
    }
}

impl Trace for DiskTrace {
    /// Store `[u8]` data in the trace.
    ///
    /// Using a string `kind` is probably a bug, but should work as long as the
    /// data is UTF-8.
    fn make_binary(&mut self, kind: DataKind, data: &[u8]) -> Data {
        self.data_id += 1;
        let name = std::format!("data{}.{}", self.data_id, kind);
        let _ = std::fs::write(self.path.join(&name), data);
        Data::File(name)
    }

    /// Store `str` data in the trace.
    ///
    /// Using a binary `kind` is fine, but it's not clear why not use
    /// `make_binary` instead.
    fn make_string(&mut self, kind: DataKind, data: &str) -> Data {
        self.make_binary(kind, data.as_bytes())
    }

    fn add(&mut self, action: Action<'_, PointerReferences>)
    where
        for<'a> Action<'a, PointerReferences>: serde::Serialize,
    {
        match ron::ser::to_string_pretty(&action, self.config.clone()) {
            Ok(string) => {
                let _ = writeln!(self.file, "{string},");
            }
            Err(e) => {
                log::warn!("RON serialization failure: {e:?}");
            }
        }
    }
}

impl Drop for DiskTrace {
    fn drop(&mut self) {
        let _ = self.file.write_all(b"]");
    }
}

#[derive(Default)]
pub struct MemoryTrace {
    actions: Vec<Action<'static, PointerReferences>>,
}

impl MemoryTrace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn actions(&self) -> &[Action<'static, PointerReferences>] {
        &self.actions
    }
}

impl Trace for MemoryTrace {
    /// Store `[u8]` data in the trace.
    ///
    /// Using a string `kind` is probably a bug, but should work as long as the
    /// data is UTF-8.
    fn make_binary(&mut self, kind: DataKind, data: &[u8]) -> Data {
        Data::Binary(kind, data.to_vec())
    }

    /// Store `str` data in the trace.
    ///
    /// Using a binary `kind` is fine, but it's not clear why not use
    /// `make_binary` instead.
    fn make_string(&mut self, kind: DataKind, data: &str) -> Data {
        Data::String(kind, data.to_string())
    }

    fn add(&mut self, action: Action<'_, PointerReferences>)
    where
        for<'a> Action<'a, PointerReferences>: serde::Serialize,
    {
        self.actions.push(action_to_owned(action))
    }
}

pub(crate) trait IntoTrace {
    type Output;
    fn into_trace(self) -> Self::Output;

    fn to_trace(&self) -> Self::Output
    where
        Self: Sized + Clone,
    {
        self.clone().into_trace()
    }
}

impl<T: StorageItem> IntoTrace for Arc<T> {
    type Output = PointerId<T::Marker>;
    fn into_trace(self) -> Self::Output {
        self.to_trace()
    }

    fn to_trace(&self) -> Self::Output {
        PointerId::from(self)
    }
}

/// This will work as expected on heap-allocated types that are not moved around.
pub(crate) unsafe fn to_trace<T: StorageItem>(t: &T) -> PointerId<T::Marker> {
    PointerId::PointerId(
        #[expect(trivial_casts)]
        core::num::NonZeroUsize::new(t as *const T as usize).unwrap(),
        PhantomData,
    )
}

impl IntoTrace for ArcCommand {
    type Output = Command<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        match self {
            ArcCommand::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => Command::CopyBufferToBuffer {
                src: src.to_trace(),
                src_offset,
                dst: dst.to_trace(),
                dst_offset,
                size,
            },
            ArcCommand::CopyBufferToTexture { src, dst, size } => Command::CopyBufferToTexture {
                src: src.into_trace(),
                dst: dst.into_trace(),
                size,
            },
            ArcCommand::CopyTextureToBuffer { src, dst, size } => Command::CopyTextureToBuffer {
                src: src.into_trace(),
                dst: dst.into_trace(),
                size,
            },
            ArcCommand::CopyTextureToTexture { src, dst, size } => Command::CopyTextureToTexture {
                src: src.into_trace(),
                dst: dst.into_trace(),
                size,
            },
            ArcCommand::ClearBuffer { dst, offset, size } => Command::ClearBuffer {
                dst: dst.to_trace(),
                offset,
                size,
            },
            ArcCommand::ClearTexture {
                dst,
                subresource_range,
            } => Command::ClearTexture {
                dst: dst.to_trace(),
                subresource_range,
            },
            ArcCommand::WriteTimestamp {
                query_set,
                query_index,
            } => Command::WriteTimestamp {
                query_set: query_set.to_trace(),
                query_index,
            },
            ArcCommand::ResolveQuerySet {
                query_set,
                start_query,
                query_count,
                destination,
                destination_offset,
            } => Command::ResolveQuerySet {
                query_set: query_set.to_trace(),
                start_query,
                query_count,
                destination: destination.to_trace(),
                destination_offset,
            },
            ArcCommand::PushDebugGroup(label) => Command::PushDebugGroup(label),
            ArcCommand::PopDebugGroup => Command::PopDebugGroup,
            ArcCommand::InsertDebugMarker(label) => Command::InsertDebugMarker(label),
            ArcCommand::RunComputePass {
                pass,
                timestamp_writes,
            } => Command::RunComputePass {
                pass: pass.into_trace(),
                timestamp_writes: timestamp_writes.map(|tw| tw.into_trace()),
            },
            ArcCommand::RunRenderPass {
                pass,
                color_attachments,
                depth_stencil_attachment,
                timestamp_writes,
                occlusion_query_set,
                multiview_mask,
            } => Command::RunRenderPass {
                pass: pass.into_trace(),
                color_attachments: color_attachments.into_trace(),
                depth_stencil_attachment: depth_stencil_attachment.map(|d| d.into_trace()),
                timestamp_writes: timestamp_writes.map(|tw| tw.into_trace()),
                occlusion_query_set: occlusion_query_set.map(|q| q.to_trace()),
                multiview_mask,
            },
            ArcCommand::BuildAccelerationStructures { blas, tlas } => {
                Command::BuildAccelerationStructures {
                    blas: blas.into_iter().map(|b| b.into_trace()).collect(),
                    tlas: tlas.into_iter().map(|b| b.into_trace()).collect(),
                }
            }
            ArcCommand::TransitionResources {
                buffer_transitions: _,
                texture_transitions: _,
            } => {
                // TransitionResources does not exist in Command, so skip or handle as needed.
                // If you want to ignore, you could panic or return a default.
                panic!("TransitionResources cannot be converted to Command");
            }
        }
    }
}

impl<T: IntoTrace> IntoTrace for wgt::TexelCopyBufferInfo<T> {
    type Output = wgt::TexelCopyBufferInfo<T::Output>;
    fn into_trace(self) -> Self::Output {
        wgt::TexelCopyBufferInfo {
            buffer: self.buffer.into_trace(),
            layout: self.layout,
        }
    }
}

impl<T: IntoTrace> IntoTrace for wgt::TexelCopyTextureInfo<T> {
    type Output = wgt::TexelCopyTextureInfo<T::Output>;
    fn into_trace(self) -> Self::Output {
        wgt::TexelCopyTextureInfo {
            texture: self.texture.into_trace(),
            mip_level: self.mip_level,
            origin: self.origin,
            aspect: self.aspect,
        }
    }
}

impl IntoTrace for ArcPassTimestampWrites {
    type Output = crate::command::PassTimestampWrites<PointerId<markers::QuerySet>>;
    fn into_trace(self) -> Self::Output {
        crate::command::PassTimestampWrites {
            query_set: self.query_set.into_trace(),
            beginning_of_pass_write_index: self.beginning_of_pass_write_index,
            end_of_pass_write_index: self.end_of_pass_write_index,
        }
    }
}

impl IntoTrace for ColorAttachments {
    type Output = ColorAttachments<PointerId<markers::TextureView>>;
    fn into_trace(self) -> Self::Output {
        self.into_iter()
            .map(|opt| {
                opt.map(|att| RenderPassColorAttachment {
                    view: att.view.into_trace(),
                    depth_slice: att.depth_slice,
                    resolve_target: att.resolve_target.map(|r| r.into_trace()),
                    load_op: att.load_op,
                    store_op: att.store_op,
                })
            })
            .collect()
    }
}

impl<TV: IntoTrace> IntoTrace for ResolvedRenderPassDepthStencilAttachment<TV> {
    type Output = ResolvedRenderPassDepthStencilAttachment<TV::Output>;
    fn into_trace(self) -> Self::Output {
        ResolvedRenderPassDepthStencilAttachment {
            view: self.view.into_trace(),
            depth: self.depth,
            stencil: self.stencil,
        }
    }
}

impl IntoTrace for crate::ray_tracing::OwnedBlasBuildEntry<ArcReferences> {
    type Output = crate::ray_tracing::OwnedBlasBuildEntry<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        crate::ray_tracing::OwnedBlasBuildEntry {
            blas: self.blas.into_trace(),
            geometries: self.geometries.into_trace(),
        }
    }
}

impl IntoTrace for crate::ray_tracing::OwnedBlasGeometries<ArcReferences> {
    type Output = crate::ray_tracing::OwnedBlasGeometries<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        match self {
            crate::ray_tracing::OwnedBlasGeometries::TriangleGeometries(geos) => {
                crate::ray_tracing::OwnedBlasGeometries::TriangleGeometries(
                    geos.into_iter().map(|g| g.into_trace()).collect(),
                )
            }
            crate::ray_tracing::OwnedBlasGeometries::AabbGeometries(geos) => {
                crate::ray_tracing::OwnedBlasGeometries::AabbGeometries(
                    geos.into_iter().map(|g| g.into_trace()).collect(),
                )
            }
        }
    }
}

impl IntoTrace for crate::ray_tracing::OwnedBlasTriangleGeometry<ArcReferences> {
    type Output = crate::ray_tracing::OwnedBlasTriangleGeometry<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        crate::ray_tracing::OwnedBlasTriangleGeometry {
            size: self.size,
            vertex_buffer: self.vertex_buffer.into_trace(),
            index_buffer: self.index_buffer.map(|b| b.into_trace()),
            transform_buffer: self.transform_buffer.map(|b| b.into_trace()),
            first_vertex: self.first_vertex,
            vertex_stride: self.vertex_stride,
            first_index: self.first_index,
            transform_buffer_offset: self.transform_buffer_offset,
        }
    }
}

impl IntoTrace for crate::ray_tracing::OwnedBlasAabbGeometry<ArcReferences> {
    type Output = crate::ray_tracing::OwnedBlasAabbGeometry<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        crate::ray_tracing::OwnedBlasAabbGeometry {
            size: self.size,
            stride: self.stride,
            aabb_buffer: self.aabb_buffer.into_trace(),
            primitive_offset: self.primitive_offset,
        }
    }
}

impl IntoTrace for crate::ray_tracing::OwnedTlasPackage<ArcReferences> {
    type Output = crate::ray_tracing::OwnedTlasPackage<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        crate::ray_tracing::OwnedTlasPackage {
            tlas: self.tlas.into_trace(),
            instances: self
                .instances
                .into_iter()
                .map(|opt| opt.map(|inst| inst.into_trace()))
                .collect(),
            lowest_unmodified: self.lowest_unmodified,
        }
    }
}

impl IntoTrace for crate::ray_tracing::OwnedTlasInstance<ArcReferences> {
    type Output = crate::ray_tracing::OwnedTlasInstance<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        crate::ray_tracing::OwnedTlasInstance {
            blas: self.blas.into_trace(),
            transform: self.transform,
            custom_data: self.custom_data,
            mask: self.mask,
        }
    }
}

impl<C: IntoTrace> IntoTrace for BasePass<C, Infallible> {
    type Output = BasePass<C::Output, Infallible>;

    fn into_trace(self) -> Self::Output {
        BasePass {
            label: self.label,
            error: self.error,
            commands: self
                .commands
                .into_iter()
                .map(|cmd| cmd.into_trace())
                .collect(),
            dynamic_offsets: self.dynamic_offsets,
            string_data: self.string_data,
            immediates_data: self.immediates_data,
        }
    }
}

impl IntoTrace for ArcComputeCommand {
    type Output = ComputeCommand<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        use ComputeCommand as C;
        match self {
            C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group: bind_group.map(|bg| bg.into_trace()),
            },
            C::SetPipeline(id) => C::SetPipeline(id.into_trace()),
            C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            },
            C::DispatchWorkgroups(groups) => C::DispatchWorkgroups(groups),
            C::DispatchWorkgroupsIndirect { buffer, offset } => C::DispatchWorkgroupsIndirect {
                buffer: buffer.into_trace(),
                offset,
            },
            C::PushDebugGroup { color, len } => C::PushDebugGroup { color, len },
            C::PopDebugGroup => C::PopDebugGroup,
            C::InsertDebugMarker { color, len } => C::InsertDebugMarker { color, len },
            C::WriteTimestamp {
                query_set,
                query_index,
            } => C::WriteTimestamp {
                query_set: query_set.into_trace(),
                query_index,
            },
            C::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => C::BeginPipelineStatisticsQuery {
                query_set: query_set.into_trace(),
                query_index,
            },
            C::EndPipelineStatisticsQuery => C::EndPipelineStatisticsQuery,
            C::TransitionResources {
                buffer_transitions,
                texture_transitions,
            } => C::TransitionResources {
                buffer_transitions: buffer_transitions
                    .into_iter()
                    .map(|buffer_transition| wgt::BufferTransition {
                        buffer: buffer_transition.buffer.into_trace(),
                        state: buffer_transition.state,
                    })
                    .collect(),
                texture_transitions: texture_transitions
                    .into_iter()
                    .map(|texture_transition| wgt::TextureTransition {
                        texture: texture_transition.texture.into_trace(),
                        selector: texture_transition.selector,
                        state: texture_transition.state,
                    })
                    .collect(),
            },
        }
    }
}

impl IntoTrace for ArcRenderCommand {
    type Output = RenderCommand<PointerReferences>;
    fn into_trace(self) -> Self::Output {
        use RenderCommand as C;
        match self {
            C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group,
            } => C::SetBindGroup {
                index,
                num_dynamic_offsets,
                bind_group: bind_group.map(|bg| bg.into_trace()),
            },
            C::SetPipeline(id) => C::SetPipeline(id.into_trace()),
            C::SetIndexBuffer {
                buffer,
                index_format,
                offset,
                size,
            } => C::SetIndexBuffer {
                buffer: buffer.into_trace(),
                index_format,
                offset,
                size,
            },
            C::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => C::SetVertexBuffer {
                slot,
                buffer: buffer.into_trace(),
                offset,
                size,
            },
            C::SetBlendConstant(color) => C::SetBlendConstant(color),
            C::SetStencilReference(val) => C::SetStencilReference(val),
            C::SetViewport {
                rect,
                depth_min,
                depth_max,
            } => C::SetViewport {
                rect,
                depth_min,
                depth_max,
            },
            C::SetScissor(rect) => C::SetScissor(rect),
            C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            } => C::SetImmediate {
                offset,
                size_bytes,
                values_offset,
            },
            C::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => C::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            },
            C::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => C::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            },
            C::DrawMeshTasks {
                group_count_x,
                group_count_y,
                group_count_z,
            } => C::DrawMeshTasks {
                group_count_x,
                group_count_y,
                group_count_z,
            },
            C::DrawIndirect {
                buffer,
                offset,
                count,
                family,
                vertex_or_index_limit,
                instance_limit,
            } => C::DrawIndirect {
                buffer: buffer.into_trace(),
                offset,
                count,
                family,
                vertex_or_index_limit,
                instance_limit,
            },
            C::MultiDrawIndirectCount {
                buffer,
                offset,
                count_buffer,
                count_buffer_offset,
                max_count,
                family,
            } => C::MultiDrawIndirectCount {
                buffer: buffer.into_trace(),
                offset,
                count_buffer: count_buffer.into_trace(),
                count_buffer_offset,
                max_count,
                family,
            },
            C::PushDebugGroup { color, len } => C::PushDebugGroup { color, len },
            C::PopDebugGroup => C::PopDebugGroup,
            C::InsertDebugMarker { color, len } => C::InsertDebugMarker { color, len },
            C::WriteTimestamp {
                query_set,
                query_index,
            } => C::WriteTimestamp {
                query_set: query_set.into_trace(),
                query_index,
            },
            C::BeginOcclusionQuery { query_index } => C::BeginOcclusionQuery { query_index },
            C::EndOcclusionQuery => C::EndOcclusionQuery,
            C::BeginPipelineStatisticsQuery {
                query_set,
                query_index,
            } => C::BeginPipelineStatisticsQuery {
                query_set: query_set.into_trace(),
                query_index,
            },
            C::EndPipelineStatisticsQuery => C::EndPipelineStatisticsQuery,
            C::ExecuteBundle(bundle) => C::ExecuteBundle(bundle.into_trace()),
        }
    }
}

impl IntoTrace for crate::binding_model::ResolvedPipelineLayoutDescriptor<'_> {
    type Output = crate::binding_model::PipelineLayoutDescriptor<
        'static,
        PointerId<markers::BindGroupLayout>,
    >;
    fn into_trace(self) -> Self::Output {
        crate::binding_model::PipelineLayoutDescriptor {
            label: self.label.map(|l| Cow::Owned(l.into_owned())),
            bind_group_layouts: self
                .bind_group_layouts
                .iter()
                .map(|bgl| bgl.to_trace())
                .collect(),
            immediate_size: self.immediate_size,
        }
    }
}

impl<'a> IntoTrace for &'_ crate::binding_model::ResolvedBindGroupDescriptor<'a> {
    type Output = TraceBindGroupDescriptor<'a>;

    fn into_trace(self) -> Self::Output {
        use crate::binding_model::{
            BindGroupEntry, BindingResource, BufferBinding, ResolvedBindingResource,
        };
        TraceBindGroupDescriptor {
            label: self.label.clone(),
            layout: self.layout.to_trace(),
            entries: Cow::Owned(
                self.entries
                    .iter()
                    .map(|entry| {
                        let resource = match &entry.resource {
                            ResolvedBindingResource::Buffer(buffer_binding) => {
                                BindingResource::Buffer(BufferBinding {
                                    buffer: buffer_binding.buffer.to_trace(),
                                    offset: buffer_binding.offset,
                                    size: buffer_binding.size,
                                })
                            }
                            ResolvedBindingResource::BufferArray(buffer_bindings) => {
                                let resolved_buffers: Vec<_> = buffer_bindings
                                    .iter()
                                    .map(|bb| BufferBinding {
                                        buffer: bb.buffer.to_trace(),
                                        offset: bb.offset,
                                        size: bb.size,
                                    })
                                    .collect();
                                BindingResource::BufferArray(Cow::Owned(resolved_buffers))
                            }
                            ResolvedBindingResource::Sampler(sampler_id) => {
                                BindingResource::Sampler(sampler_id.to_trace())
                            }
                            ResolvedBindingResource::SamplerArray(sampler_ids) => {
                                let resolved: Vec<_> =
                                    sampler_ids.iter().map(|id| id.to_trace()).collect();
                                BindingResource::SamplerArray(Cow::Owned(resolved))
                            }
                            ResolvedBindingResource::TextureView(texture_view_id) => {
                                BindingResource::TextureView(texture_view_id.to_trace())
                            }
                            ResolvedBindingResource::TextureViewArray(texture_view_ids) => {
                                let resolved: Vec<_> =
                                    texture_view_ids.iter().map(|id| id.to_trace()).collect();
                                BindingResource::TextureViewArray(Cow::Owned(resolved))
                            }
                            ResolvedBindingResource::AccelerationStructure(tlas_id) => {
                                BindingResource::AccelerationStructure(tlas_id.to_trace())
                            }
                            ResolvedBindingResource::AccelerationStructureArray(tlas_ids) => {
                                let resolved: Vec<_> =
                                    tlas_ids.iter().map(|id| id.to_trace()).collect();
                                BindingResource::AccelerationStructureArray(Cow::Owned(resolved))
                            }
                            ResolvedBindingResource::ExternalTexture(external_texture_id) => {
                                BindingResource::ExternalTexture(external_texture_id.to_trace())
                            }
                        };
                        BindGroupEntry {
                            binding: entry.binding,
                            resource,
                        }
                    })
                    .collect(),
            ),
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedGeneralRenderPipelineDescriptor<'a> {
    type Output = TraceGeneralRenderPipelineDescriptor<'a>;

    fn into_trace(self) -> Self::Output {
        TraceGeneralRenderPipelineDescriptor {
            label: self.label,
            layout: self.layout.into_trace(),
            vertex: self.vertex.into_trace(),
            primitive: self.primitive,
            depth_stencil: self.depth_stencil,
            multisample: self.multisample,
            fragment: self.fragment.map(|f| f.into_trace()),
            multiview_mask: self.multiview_mask,
            cache: self.cache.map(|c| c.into_trace()),
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedComputePipelineDescriptor<'a> {
    type Output = TraceComputePipelineDescriptor<'a>;

    fn into_trace(self) -> Self::Output {
        TraceComputePipelineDescriptor {
            label: self.label,
            layout: self.layout.into_trace(),
            stage: self.stage.into_trace(),
            cache: self.cache.map(|c| c.into_trace()),
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedProgrammableStageDescriptor<'a> {
    type Output =
        crate::pipeline::ProgrammableStageDescriptor<'a, PointerId<markers::ShaderModule>>;
    fn into_trace(self) -> Self::Output {
        crate::pipeline::ProgrammableStageDescriptor {
            module: self.module.into_trace(),
            entry_point: self.entry_point,
            constants: self.constants,
            zero_initialize_workgroup_memory: self.zero_initialize_workgroup_memory,
        }
    }
}

impl<'a> IntoTrace
    for crate::pipeline::RenderPipelineVertexProcessor<'a, Arc<crate::pipeline::ShaderModule>>
{
    type Output =
        crate::pipeline::RenderPipelineVertexProcessor<'a, PointerId<markers::ShaderModule>>;
    fn into_trace(self) -> Self::Output {
        match self {
            crate::pipeline::RenderPipelineVertexProcessor::Vertex(vertex) => {
                crate::pipeline::RenderPipelineVertexProcessor::Vertex(vertex.into_trace())
            }
            crate::pipeline::RenderPipelineVertexProcessor::Mesh(task, mesh) => {
                crate::pipeline::RenderPipelineVertexProcessor::Mesh(
                    task.map(|t| t.into_trace()),
                    mesh.into_trace(),
                )
            }
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedTaskState<'a> {
    type Output = crate::pipeline::TaskState<'a, PointerId<markers::ShaderModule>>;
    fn into_trace(self) -> Self::Output {
        crate::pipeline::TaskState {
            stage: self.stage.into_trace(),
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedMeshState<'a> {
    type Output = crate::pipeline::MeshState<'a, PointerId<markers::ShaderModule>>;
    fn into_trace(self) -> Self::Output {
        crate::pipeline::MeshState {
            stage: self.stage.into_trace(),
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedVertexState<'a> {
    type Output = crate::pipeline::VertexState<'a, PointerId<markers::ShaderModule>>;
    fn into_trace(self) -> Self::Output {
        crate::pipeline::VertexState {
            stage: self.stage.into_trace(),
            buffers: self.buffers,
        }
    }
}

impl<'a> IntoTrace for crate::pipeline::ResolvedFragmentState<'a> {
    type Output = crate::pipeline::FragmentState<'a, PointerId<markers::ShaderModule>>;
    fn into_trace(self) -> Self::Output {
        crate::pipeline::FragmentState {
            stage: self.stage.into_trace(),
            targets: self.targets,
        }
    }
}

impl<T: IntoTrace> IntoTrace for Option<T> {
    type Output = Option<T::Output>;
    fn into_trace(self) -> Self::Output {
        self.map(|v| v.into_trace())
    }
}

/// Return a copy of [`Action`] with `'static` lifetime.
///
/// This is used for in-memory tracing.
fn action_to_owned(action: Action<'_, PointerReferences>) -> Action<'static, PointerReferences> {
    use Action as A;
    match action {
        A::Init { desc, backend } => A::Init {
            desc: desc.map_label(owned_label),
            backend,
        },
        A::ConfigureSurface(surface, config) => A::ConfigureSurface(surface, config),
        A::CreateBuffer(buffer, desc) => A::CreateBuffer(buffer, desc.map_label(owned_label)),
        A::DestroyBuffer(buffer) => A::DestroyBuffer(buffer),
        A::DropBuffer(buffer) => A::DropBuffer(buffer),
        A::DestroyTexture(texture) => A::DestroyTexture(texture),
        A::DropTexture(texture) => A::DropTexture(texture),
        A::DropTextureView(texture_view) => A::DropTextureView(texture_view),
        A::DestroyExternalTexture(external_texture) => A::DestroyExternalTexture(external_texture),
        A::DropExternalTexture(external_texture) => A::DropExternalTexture(external_texture),
        A::DropSampler(sampler) => A::DropSampler(sampler),
        A::GetSurfaceTexture { id, parent } => A::GetSurfaceTexture { id, parent },
        A::Present(surface) => A::Present(surface),
        A::DiscardSurfaceTexture(surface) => A::DiscardSurfaceTexture(surface),
        A::ReleaseSurfaceTexture(surface) => A::ReleaseSurfaceTexture(surface),
        A::DropBindGroupLayout(layout) => A::DropBindGroupLayout(layout),
        A::GetRenderPipelineBindGroupLayout {
            id,
            pipeline,
            index,
        } => A::GetRenderPipelineBindGroupLayout {
            id,
            pipeline,
            index,
        },
        A::GetComputePipelineBindGroupLayout {
            id,
            pipeline,
            index,
        } => A::GetComputePipelineBindGroupLayout {
            id,
            pipeline,
            index,
        },
        A::DropPipelineLayout(layout) => A::DropPipelineLayout(layout),
        A::DropBindGroup(bind_group) => A::DropBindGroup(bind_group),
        A::DropShaderModule(shader_module) => A::DropShaderModule(shader_module),
        A::DropComputePipeline(pipeline) => A::DropComputePipeline(pipeline),
        A::DropRenderPipeline(pipeline) => A::DropRenderPipeline(pipeline),
        A::DropPipelineCache(cache) => A::DropPipelineCache(cache),
        A::DropRenderBundle(render_bundle) => A::DropRenderBundle(render_bundle),
        A::DestroyQuerySet(query_set) => A::DestroyQuerySet(query_set),
        A::DropQuerySet(query_set) => A::DropQuerySet(query_set),
        A::WriteBuffer {
            id,
            data,
            offset,
            size,
            queued,
        } => A::WriteBuffer {
            id,
            data,
            offset,
            size,
            queued,
        },
        A::WriteTexture {
            to,
            data,
            layout,
            size,
        } => A::WriteTexture {
            to,
            data,
            layout,
            size,
        },
        A::Submit(index, commands) => A::Submit(index, commands),
        A::FailedCommands {
            commands,
            failed_at_submit,
            error,
        } => A::FailedCommands {
            commands,
            failed_at_submit,
            error,
        },
        A::DropBlas(blas) => A::DropBlas(blas),
        A::DropTlas(tlas) => A::DropTlas(tlas),

        A::CreateTexture(id, desc) => A::CreateTexture(id, desc.map_label(owned_label)),
        A::CreateTextureError(id, desc) => A::CreateTextureError(id, desc.map_label(owned_label)),
        A::CreateTextureView { id, parent, desc } => A::CreateTextureView {
            id,
            parent,
            desc: crate::resource::TextureViewDescriptor {
                label: owned_label(&desc.label),
                format: desc.format,
                dimension: desc.dimension,
                usage: desc.usage,
                range: desc.range,
            },
        },
        A::CreateExternalTexture { id, desc, planes } => A::CreateExternalTexture {
            id,
            desc: desc.map_label(owned_label),
            planes,
        },
        A::CreateSampler(id, desc) => A::CreateSampler(
            id,
            crate::resource::SamplerDescriptor {
                label: owned_label(&desc.label),
                address_modes: desc.address_modes,
                mag_filter: desc.mag_filter,
                min_filter: desc.min_filter,
                mipmap_filter: desc.mipmap_filter,
                lod_min_clamp: desc.lod_min_clamp,
                lod_max_clamp: desc.lod_max_clamp,
                compare: desc.compare,
                anisotropy_clamp: desc.anisotropy_clamp,
                border_color: desc.border_color,
            },
        ),
        A::CreateBindGroupLayout(id, desc) => A::CreateBindGroupLayout(
            id,
            crate::binding_model::BindGroupLayoutDescriptor {
                label: owned_label(&desc.label),
                entries: Cow::Owned(desc.entries.into_owned()),
            },
        ),
        A::CreatePipelineLayout(id, desc) => A::CreatePipelineLayout(
            id,
            crate::binding_model::PipelineLayoutDescriptor {
                label: owned_label(&desc.label),
                bind_group_layouts: Cow::Owned(desc.bind_group_layouts.into_owned()),
                immediate_size: desc.immediate_size,
            },
        ),
        A::CreateBindGroup(id, desc) => A::CreateBindGroup(
            id,
            crate::binding_model::BindGroupDescriptor {
                label: owned_label(&desc.label),
                layout: desc.layout,
                entries: desc
                    .entries
                    .iter()
                    .map(|e| crate::binding_model::BindGroupEntry {
                        binding: e.binding,
                        resource: match &e.resource {
                            crate::binding_model::BindingResource::Buffer(buffer_binding) => {
                                crate::binding_model::BindingResource::Buffer(
                                    buffer_binding.clone(),
                                )
                            }
                            crate::binding_model::BindingResource::BufferArray(cow) => {
                                crate::binding_model::BindingResource::BufferArray(Cow::Owned(
                                    cow.clone().into_owned(),
                                ))
                            }
                            crate::binding_model::BindingResource::Sampler(sampler) => {
                                crate::binding_model::BindingResource::Sampler(*sampler)
                            }
                            crate::binding_model::BindingResource::SamplerArray(cow) => {
                                crate::binding_model::BindingResource::SamplerArray(Cow::Owned(
                                    cow.clone().into_owned(),
                                ))
                            }
                            crate::binding_model::BindingResource::TextureView(texture_view) => {
                                crate::binding_model::BindingResource::TextureView(*texture_view)
                            }
                            crate::binding_model::BindingResource::TextureViewArray(cow) => {
                                crate::binding_model::BindingResource::TextureViewArray(Cow::Owned(
                                    cow.clone().into_owned(),
                                ))
                            }
                            crate::binding_model::BindingResource::AccelerationStructure(
                                acceleration_structure,
                            ) => crate::binding_model::BindingResource::AccelerationStructure(
                                *acceleration_structure,
                            ),
                            crate::binding_model::BindingResource::AccelerationStructureArray(
                                cow,
                            ) => crate::binding_model::BindingResource::AccelerationStructureArray(
                                Cow::Owned(cow.clone().into_owned()),
                            ),
                            crate::binding_model::BindingResource::ExternalTexture(
                                external_texture,
                            ) => crate::binding_model::BindingResource::ExternalTexture(
                                *external_texture,
                            ),
                        },
                    })
                    .collect(),
            },
        ),
        A::CreateShaderModule { id, desc, data } => A::CreateShaderModule {
            id,
            desc: crate::pipeline::ShaderModuleDescriptor {
                label: owned_label(&desc.label),
                runtime_checks: desc.runtime_checks,
            },
            data,
        },
        A::CreateShaderModulePassthrough {
            id,
            data,
            label,
            entry_points,
        } => A::CreateShaderModulePassthrough {
            id,
            data,
            label: owned_label(&label),
            entry_points: entry_points
                .iter()
                .map(|ep| wgt::PassthroughShaderEntryPoint {
                    name: Cow::Owned(ep.name.to_string()),
                    workgroup_size: ep.workgroup_size,
                })
                .collect(),
        },
        A::CreateComputePipeline { id, desc } => A::CreateComputePipeline {
            id,
            desc: crate::pipeline::ComputePipelineDescriptor {
                label: owned_label(&desc.label),
                layout: desc.layout,
                stage: owned_stage(desc.stage),
                cache: desc.cache,
            },
        },
        A::CreateGeneralRenderPipeline { id, desc } => A::CreateGeneralRenderPipeline {
            id,
            desc: crate::pipeline::GeneralRenderPipelineDescriptor {
                label: owned_label(&desc.label),
                layout: desc.layout,
                vertex: match desc.vertex {
                    crate::pipeline::RenderPipelineVertexProcessor::Vertex(
                        crate::pipeline::VertexState { stage, buffers },
                    ) => crate::pipeline::RenderPipelineVertexProcessor::Vertex(
                        crate::pipeline::VertexState {
                            stage: owned_stage(stage),
                            buffers: buffers
                                .iter()
                                .map(|b| {
                                    b.clone().map(|buffer| crate::pipeline::VertexBufferLayout {
                                        array_stride: buffer.array_stride,
                                        step_mode: buffer.step_mode,
                                        attributes: Cow::Owned(buffer.attributes.into_owned()),
                                    })
                                })
                                .collect(),
                        },
                    ),
                    crate::pipeline::RenderPipelineVertexProcessor::Mesh(task, mesh) => {
                        crate::pipeline::RenderPipelineVertexProcessor::Mesh(
                            task.map(|t| crate::pipeline::TaskState {
                                stage: owned_stage(t.stage),
                            }),
                            crate::pipeline::MeshState {
                                stage: owned_stage(mesh.stage),
                            },
                        )
                    }
                },
                primitive: desc.primitive,
                depth_stencil: desc.depth_stencil,
                multisample: desc.multisample,
                fragment: desc.fragment.map(|f| crate::pipeline::FragmentState {
                    stage: owned_stage(f.stage),
                    targets: Cow::Owned(f.targets.into_owned()),
                }),
                multiview_mask: desc.multiview_mask,
                cache: desc.cache,
            },
        },
        A::CreatePipelineCache { id, desc } => A::CreatePipelineCache {
            id,
            desc: crate::pipeline::PipelineCacheDescriptor {
                label: owned_label(&desc.label),
                data: desc.data.map(|d| Cow::Owned(d.to_vec())),
                fallback: desc.fallback,
            },
        },
        A::CreateRenderBundle { id, desc, base } => A::CreateRenderBundle {
            id,
            desc: crate::command::RenderBundleEncoderDescriptor {
                label: owned_label(&desc.label),
                color_formats: Cow::Owned(desc.color_formats.into_owned()),
                depth_stencil: desc.depth_stencil,
                sample_count: desc.sample_count,
                multiview: desc.multiview,
            },
            base,
        },
        A::CreateQuerySet { id, desc } => A::CreateQuerySet {
            id,
            desc: desc.map_label(owned_label),
        },
        A::CreateBlas { id, desc, sizes } => A::CreateBlas {
            id,
            desc: desc.map_label(owned_label),
            sizes,
        },
        A::CreateTlas { id, desc } => A::CreateTlas {
            id,
            desc: desc.map_label(owned_label),
        },
    }
}

fn owned_stage<SM>(
    stage: crate::pipeline::ProgrammableStageDescriptor<'_, SM>,
) -> crate::pipeline::ProgrammableStageDescriptor<'static, SM> {
    crate::pipeline::ProgrammableStageDescriptor {
        module: stage.module,
        entry_point: owned_label(&stage.entry_point),
        constants: stage.constants,
        zero_initialize_workgroup_memory: stage.zero_initialize_workgroup_memory,
    }
}

fn owned_label(l: &Option<Cow<'_, str>>) -> Option<Cow<'static, str>> {
    l.as_ref().map(|l| Cow::Owned(l.to_string()))
}
