use wgpu_core::command::{CommandEncoderError, EncoderStateError};
use wgpu_core::ray_tracing::{
    BlasAabbGeometry, BlasBuildEntry, BlasGeometries, BlasTriangleGeometry, TlasInstance,
    TlasPackage,
};
use wgpu_core::Label;
use wgt::{BufferAddress, Extent3d, ImageSubresourceRange};

use crate::global::Global;
use crate::id::{BlasId, BufferId, CommandEncoderId, TextureId, TlasId};
use crate::{id, TexelCopyBufferInfo};

impl Global {
    /// Finishes a command encoder, creating a command buffer and returning errors that were
    /// deferred until now.
    ///
    /// The returned `String` is the label of the command encoder, supplied so that `wgpu` can
    /// include the label when printing deferred errors without having its own copy of the label.
    /// This is a kludge and should be replaced if we think of a better solution to propagating
    /// labels.
    pub fn command_encoder_finish(
        &self,
        encoder_id: CommandEncoderId,
        desc: &wgt::CommandBufferDescriptor<Label>,
        id_in: Option<id::CommandBufferId>,
    ) -> (id::CommandBufferId, Option<(String, CommandEncoderError)>) {
        let hub = &self.hub;
        let cmd_enc = hub.command_encoders.get(encoder_id);

        let (cmd_buf, opt_error) = cmd_enc.finish(desc);
        let cmd_buf_id = hub.command_buffers.prepare(id_in).assign(cmd_buf);

        (cmd_buf_id, opt_error)
    }

    pub fn command_encoder_push_debug_group(
        &self,
        encoder_id: CommandEncoderId,
        label: &str,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);
        cmd_enc.push_debug_group(label)
    }

    pub fn command_encoder_insert_debug_marker(
        &self,
        encoder_id: CommandEncoderId,
        label: &str,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);
        cmd_enc.insert_debug_marker(label)
    }

    pub fn command_encoder_pop_debug_group(
        &self,
        encoder_id: CommandEncoderId,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);
        cmd_enc.pop_debug_group()
    }
}

impl Global {
    pub fn command_encoder_clear_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        cmd_enc.clear_buffer(hub.buffers.get(dst), offset, size)
    }

    pub fn command_encoder_clear_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: TextureId,
        subresource_range: &ImageSubresourceRange,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);

        cmd_enc.clear_texture(hub.textures.get(dst), subresource_range)
    }
}

impl Global {
    pub fn command_encoder_write_timestamp(
        &self,
        command_encoder_id: CommandEncoderId,
        query_set_id: id::QuerySetId,
        query_index: u32,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        cmd_enc.write_timestamp(hub.query_sets.get(query_set_id), query_index)
    }

    pub fn command_encoder_resolve_query_set(
        &self,
        command_encoder_id: CommandEncoderId,
        query_set_id: id::QuerySetId,
        start_query: u32,
        query_count: u32,
        destination: BufferId,
        destination_offset: BufferAddress,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);

        cmd_enc.resolve_query_set(
            hub.query_sets.get(query_set_id),
            start_query,
            query_count,
            hub.buffers.get(destination),
            destination_offset,
        )
    }
}

impl Global {
    pub fn command_encoder_mark_acceleration_structures_built(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_ids: &[BlasId],
        tlas_ids: &[TlasId],
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);

        let blases = blas_ids
            .iter()
            .map(|&id| hub.blas_s.get(id))
            .collect::<Vec<_>>();
        let tlases = tlas_ids
            .iter()
            .map(|&id| hub.tlas_s.get(id))
            .collect::<Vec<_>>();
        cmd_enc.mark_acceleration_structures_built(&blases, &tlases)
    }

    pub fn command_encoder_build_acceleration_structures<'a>(
        &self,
        command_encoder_id: CommandEncoderId,
        blas_iter: impl Iterator<Item = BlasBuildEntry<'a, BlasId, BufferId>>,
        tlas_iter: impl Iterator<Item = TlasPackage<'a, TlasId, BlasId>>,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let blases = blas_iter.map(|e| BlasBuildEntry {
            blas: hub.blas_s.get(e.blas),
            geometries: match e.geometries {
                BlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let triangle_geometries = triangle_geometries.map(|tg| BlasTriangleGeometry {
                        size: tg.size,
                        vertex_buffer: hub.buffers.get(tg.vertex_buffer),
                        index_buffer: tg
                            .index_buffer
                            .map(|index_buffer| hub.buffers.get(index_buffer)),
                        transform_buffer: tg
                            .transform_buffer
                            .map(|transform_buffer| hub.buffers.get(transform_buffer)),
                        first_vertex: tg.first_vertex,
                        vertex_stride: tg.vertex_stride,
                        first_index: tg.first_index,
                        transform_buffer_offset: tg.transform_buffer_offset,
                    });
                    BlasGeometries::TriangleGeometries(Box::new(triangle_geometries.into_iter()))
                }
                BlasGeometries::AabbGeometries(aabb_geometries) => {
                    let aabb_geometries = aabb_geometries.map(|ag| BlasAabbGeometry {
                        size: ag.size,
                        stride: ag.stride,
                        aabb_buffer: hub.buffers.get(ag.aabb_buffer),
                        primitive_offset: ag.primitive_offset,
                    });
                    BlasGeometries::AabbGeometries(Box::new(aabb_geometries.into_iter()))
                }
            },
        });
        let tlases = tlas_iter.map(|e| TlasPackage {
            tlas: hub.tlas_s.get(e.tlas),
            instances: Box::new(e.instances.map(|instance| {
                instance.as_ref().map(|instance| TlasInstance {
                    blas: hub.blas_s.get(instance.blas),
                    transform: instance.transform,
                    custom_data: instance.custom_data,
                    mask: instance.mask,
                })
            })),
            lowest_unmodified: e.lowest_unmodified,
        });
        cmd_enc.build_acceleration_structures(blases.into_iter(), tlases.into_iter())
    }
}

impl Global {
    pub fn command_encoder_copy_buffer_to_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        source: BufferId,
        source_offset: BufferAddress,
        destination: BufferId,
        destination_offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let source = self.resolve_buffer_id(source);
        let destination = self.resolve_buffer_id(destination);
        cmd_enc.copy_buffer_to_buffer(source, source_offset, destination, destination_offset, size)
    }

    pub fn command_encoder_copy_buffer_to_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyBufferInfo,
        destination: &wgt::TexelCopyTextureInfo<TextureId>,
        copy_size: &Extent3d,
    ) -> Result<(), EncoderStateError> {
        let cmd_enc = self.hub.command_encoders.get(command_encoder_id);
        let source = wgt::TexelCopyBufferInfo {
            buffer: self.resolve_buffer_id(source.buffer),
            layout: source.layout,
        };
        let destination = wgt::TexelCopyTextureInfo {
            texture: self.resolve_texture_id(destination.texture),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };
        cmd_enc.copy_buffer_to_texture(&source, &destination, copy_size)
    }

    pub fn command_encoder_copy_texture_to_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &wgt::TexelCopyTextureInfo<TextureId>,
        destination: &TexelCopyBufferInfo,
        copy_size: &Extent3d,
    ) -> Result<(), EncoderStateError> {
        let cmd_enc = self.hub.command_encoders.get(command_encoder_id);

        let source = wgt::TexelCopyTextureInfo {
            texture: self.resolve_texture_id(source.texture),
            mip_level: source.mip_level,
            origin: source.origin,
            aspect: source.aspect,
        };
        let destination = wgt::TexelCopyBufferInfo {
            buffer: self.resolve_buffer_id(destination.buffer),
            layout: destination.layout,
        };
        cmd_enc.copy_texture_to_buffer(&source, &destination, copy_size)
    }

    pub fn command_encoder_copy_texture_to_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &wgt::TexelCopyTextureInfo<TextureId>,
        destination: &wgt::TexelCopyTextureInfo<TextureId>,
        copy_size: &Extent3d,
    ) -> Result<(), EncoderStateError> {
        let cmd_enc = self.hub.command_encoders.get(command_encoder_id);

        let source = wgt::TexelCopyTextureInfo {
            texture: self.resolve_texture_id(source.texture),
            mip_level: source.mip_level,
            origin: source.origin,
            aspect: source.aspect,
        };
        let destination = wgt::TexelCopyTextureInfo {
            texture: self.resolve_texture_id(destination.texture),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };
        cmd_enc.copy_texture_to_texture(&source, &destination, copy_size)
    }
}

impl Global {
    pub fn command_encoder_transition_resources(
        &self,
        command_encoder_id: CommandEncoderId,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<BufferId>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<TextureId>>,
    ) -> Result<(), EncoderStateError> {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let buffer_transitions = buffer_transitions
            .map(|t| {
                let buffer = hub.buffers.get(t.buffer);
                wgt::BufferTransition {
                    buffer,
                    state: t.state,
                }
            })
            .collect::<Vec<_>>();
        let texture_transitions = texture_transitions
            .map(|t| {
                let texture = hub.textures.get(t.texture);
                wgt::TextureTransition {
                    texture,
                    selector: t.selector,
                    state: t.state,
                }
            })
            .collect::<Vec<_>>();
        cmd_enc.transition_resources(
            buffer_transitions.into_iter(),
            texture_transitions.into_iter(),
        )
    }
}
