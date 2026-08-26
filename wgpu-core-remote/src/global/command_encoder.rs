use wgpu_core_remote_types::encoders::{
    CommandBufferDescriptor, CommandEncoderCommand, DebugCommand, TexelCopyBufferInfo,
    TexelCopyTextureInfo,
};
use wgt::{BufferAddress, Extent3d, ImageSubresourceRange};

use crate::global::Global;
use crate::hub::Hub;
use crate::id;
use crate::id::{BufferId, CommandEncoderId, TextureId};

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
        desc: &CommandBufferDescriptor,
        id_in: id::CommandBufferId,
    ) {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            command_encoders,
            command_buffers,
            ..
        } = &mut *hub;
        let cmd_enc = command_encoders.get(encoder_id);

        let cmd_buf = cmd_enc.finish(desc);
        command_buffers.assign(id_in, cmd_buf);
    }

    pub fn command_encoder_push_debug_group(&self, encoder_id: CommandEncoderId, label: &str) {
        let hub = self.hub.borrow();

        let cmd_enc = hub.command_encoders.get(encoder_id);
        cmd_enc.push_debug_group(label)
    }

    pub fn command_encoder_insert_debug_marker(&self, encoder_id: CommandEncoderId, label: &str) {
        let hub = self.hub.borrow();

        let cmd_enc = hub.command_encoders.get(encoder_id);
        cmd_enc.insert_debug_marker(label)
    }

    pub fn command_encoder_pop_debug_group(&self, encoder_id: CommandEncoderId) {
        let hub = self.hub.borrow();

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
    ) {
        let hub = self.hub.borrow();

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        cmd_enc.clear_buffer(hub.buffers.get(dst), offset, size)
    }

    pub fn command_encoder_clear_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        dst: TextureId,
        subresource_range: &ImageSubresourceRange,
    ) {
        let hub = self.hub.borrow();

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
    ) {
        let hub = self.hub.borrow();

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
    ) {
        let hub = self.hub.borrow();

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
    pub fn command_encoder_copy_buffer_to_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        source: BufferId,
        source_offset: BufferAddress,
        destination: BufferId,
        destination_offset: BufferAddress,
        size: Option<BufferAddress>,
    ) {
        let hub = self.hub.borrow();

        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let source = hub.buffers.get(source);
        let destination = hub.buffers.get(destination);
        cmd_enc.copy_buffer_to_buffer(source, source_offset, destination, destination_offset, size)
    }

    pub fn command_encoder_copy_buffer_to_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyBufferInfo,
        destination: &TexelCopyTextureInfo,
        copy_size: &Extent3d,
    ) {
        let hub = self.hub.borrow();
        let cmd_enc = hub.command_encoders.get(command_encoder_id);
        let source = wgt::TexelCopyBufferInfo {
            buffer: hub.buffers.get(source.buffer),
            layout: source.layout,
        };
        let destination = wgt::TexelCopyTextureInfo {
            texture: hub.textures.get(destination.texture),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };
        cmd_enc.copy_buffer_to_texture(&source, &destination, copy_size)
    }

    pub fn command_encoder_copy_texture_to_buffer(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyTextureInfo,
        destination: &TexelCopyBufferInfo,
        copy_size: &Extent3d,
    ) {
        let hub = self.hub.borrow();
        let cmd_enc = hub.command_encoders.get(command_encoder_id);

        let source = wgt::TexelCopyTextureInfo {
            texture: hub.textures.get(source.texture),
            mip_level: source.mip_level,
            origin: source.origin,
            aspect: source.aspect,
        };
        let destination = wgt::TexelCopyBufferInfo {
            buffer: hub.buffers.get(destination.buffer),
            layout: destination.layout,
        };
        cmd_enc.copy_texture_to_buffer(&source, &destination, copy_size)
    }

    pub fn command_encoder_copy_texture_to_texture(
        &self,
        command_encoder_id: CommandEncoderId,
        source: &TexelCopyTextureInfo,
        destination: &TexelCopyTextureInfo,
        copy_size: &Extent3d,
    ) {
        let hub = self.hub.borrow();
        let cmd_enc = hub.command_encoders.get(command_encoder_id);

        let source = wgt::TexelCopyTextureInfo {
            texture: hub.textures.get(source.texture),
            mip_level: source.mip_level,
            origin: source.origin,
            aspect: source.aspect,
        };
        let destination = wgt::TexelCopyTextureInfo {
            texture: hub.textures.get(destination.texture),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };
        cmd_enc.copy_texture_to_texture(&source, &destination, copy_size)
    }
}

impl Global {
    pub fn handle_command_encoder_command<'a>(
        &self,
        command_encoder_id: CommandEncoderId,
        command: CommandEncoderCommand<'a>,
    ) {
        match command {
            CommandEncoderCommand::BeginRenderPass {
                desc,
                render_pass_encoder_id,
            } => self.command_encoder_begin_render_pass(
                command_encoder_id,
                &desc,
                render_pass_encoder_id,
            ),
            CommandEncoderCommand::BeginComputePass {
                desc,
                compute_pass_encoder_id,
            } => self.command_encoder_begin_compute_pass(
                command_encoder_id,
                &desc,
                compute_pass_encoder_id,
            ),
            CommandEncoderCommand::CopyBufferToBuffer {
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => self.command_encoder_copy_buffer_to_buffer(
                command_encoder_id,
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            ),
            CommandEncoderCommand::CopyBufferToTexture {
                source,
                destination,
                copy_size,
            } => self.command_encoder_copy_buffer_to_texture(
                command_encoder_id,
                &source,
                &destination,
                &copy_size,
            ),
            CommandEncoderCommand::CopyTextureToBuffer {
                source,
                destination,
                copy_size,
            } => self.command_encoder_copy_texture_to_buffer(
                command_encoder_id,
                &source,
                &destination,
                &copy_size,
            ),
            CommandEncoderCommand::CopyTextureToTexture {
                source,
                destination,
                copy_size,
            } => self.command_encoder_copy_texture_to_texture(
                command_encoder_id,
                &source,
                &destination,
                &copy_size,
            ),
            CommandEncoderCommand::ClearBuffer {
                buffer,
                offset,
                size,
            } => self.command_encoder_clear_buffer(command_encoder_id, buffer, offset, size),
            CommandEncoderCommand::ResolveQuerySet {
                query_set,
                first_query,
                query_count,
                destination,
                destination_offset,
            } => self.command_encoder_resolve_query_set(
                command_encoder_id,
                query_set,
                first_query,
                query_count,
                destination,
                destination_offset,
            ),
            CommandEncoderCommand::DebugCommand(debug_command) => match debug_command {
                DebugCommand::PushDebugGroup(label) => {
                    self.command_encoder_push_debug_group(command_encoder_id, &label)
                }
                DebugCommand::PopDebugGroup => {
                    self.command_encoder_pop_debug_group(command_encoder_id)
                }
                DebugCommand::InsertDebugMarker(label) => {
                    self.command_encoder_insert_debug_marker(command_encoder_id, &label)
                }
            },
            CommandEncoderCommand::Finish {
                desc,
                command_buffer_id,
            } => self.command_encoder_finish(command_encoder_id, &desc, command_buffer_id),
        }
    }
}
