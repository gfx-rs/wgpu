/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    cow_label, identity::IdentityRecyclerFactory, AdapterInformation, ByteBuf,
    CommandEncoderAction, DeviceAction, DropAction, QueueWriteAction, RawString,
    ShaderModuleSource, TextureAction,
};

use wgc::{gfx_select, id};

use std::{error::Error, os::raw::c_char, ptr, slice};

#[repr(C)]
pub struct ErrorBuffer {
    string: *mut c_char,
    capacity: usize,
}

impl ErrorBuffer {
    fn init(&mut self, error: impl Error) {
        use std::fmt::Write;

        let mut string = format!("{}", error);
        let mut e = error.source();
        while let Some(source) = e {
            write!(string, ", caused by: {}", source).unwrap();
            e = source.source();
        }

        assert_ne!(self.capacity, 0);
        let length = if string.len() >= self.capacity {
            log::warn!(
                "Error length {} reached capacity {}",
                string.len(),
                self.capacity
            );
            self.capacity - 1
        } else {
            string.len()
        };
        unsafe {
            ptr::copy_nonoverlapping(string.as_ptr(), self.string as *mut u8, length);
            *self.string.add(length) = 0;
        }
    }
}

// hide wgc's global in private
pub struct Global(wgc::hub::Global<IdentityRecyclerFactory>);

impl std::ops::Deref for Global {
    type Target = wgc::hub::Global<IdentityRecyclerFactory>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_new(factory: IdentityRecyclerFactory) -> *mut Global {
    log::info!("Initializing WGPU server");
    let global = Global(wgc::hub::Global::new(
        "wgpu",
        factory,
        wgt::BackendBit::PRIMARY,
    ));
    Box::into_raw(Box::new(global))
}

/// # Safety
///
/// This function is unsafe because improper use may lead to memory
/// problems. For example, a double-free may occur if the function is called
/// twice on the same raw pointer.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_delete(global: *mut Global) {
    log::info!("Terminating WGPU server");
    let _ = Box::from_raw(global);
}

#[no_mangle]
pub extern "C" fn wgpu_server_poll_all_devices(global: &Global, force_wait: bool) {
    global.poll_all_devices(force_wait).unwrap();
}

/// Request an adapter according to the specified options.
/// Provide the list of IDs to pick from.
///
/// Returns the index in this list, or -1 if unable to pick.
///
/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `id_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_instance_request_adapter(
    global: &Global,
    desc: &wgc::instance::RequestAdapterOptions,
    ids: *const id::AdapterId,
    id_length: usize,
    mut error_buf: ErrorBuffer,
) -> i8 {
    let ids = slice::from_raw_parts(ids, id_length);
    match global.request_adapter(
        desc,
        wgc::instance::AdapterInputs::IdSet(ids, |i| i.backend()),
    ) {
        Ok(id) => ids.iter().position(|&i| i == id).unwrap() as i8,
        Err(e) => {
            error_buf.init(e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_adapter_pack_info(
    global: &Global,
    self_id: Option<id::AdapterId>,
    byte_buf: &mut ByteBuf,
) {
    let mut data = Vec::new();
    match self_id {
        Some(id) => {
            let raw_info = gfx_select!(id => global.adapter_get_info(id)).unwrap();
            let info = AdapterInformation {
                id,
                ty: raw_info.device_type,
                limits: gfx_select!(id => global.adapter_limits(id)).unwrap(),
                features: gfx_select!(id => global.adapter_features(id)).unwrap(),
            };
            bincode::serialize_into(&mut data, &info).unwrap();
        }
        None => {
            bincode::serialize_into(&mut data, &0u64).unwrap();
        }
    }
    *byte_buf = ByteBuf::from_vec(data);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_adapter_request_device(
    global: &Global,
    self_id: id::AdapterId,
    byte_buf: &ByteBuf,
    new_id: id::DeviceId,
    mut error_buf: ErrorBuffer,
) {
    let desc: wgc::device::DeviceDescriptor = bincode::deserialize(byte_buf.as_slice()).unwrap();
    let trace_string = std::env::var("WGPU_TRACE").ok();
    let trace_path = trace_string
        .as_ref()
        .map(|string| std::path::Path::new(string.as_str()));
    let (_, error) =
        gfx_select!(self_id => global.adapter_request_device(self_id, &desc, trace_path, new_id));
    if let Some(err) = error {
        error_buf.init(err);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_drop(global: &Global, adapter_id: id::AdapterId) {
    gfx_select!(adapter_id => global.adapter_drop(adapter_id))
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_drop(global: &Global, self_id: id::DeviceId) {
    gfx_select!(self_id => global.device_drop(self_id))
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_buffer(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::BufferDescriptor<RawString>,
    new_id: id::BufferId,
    mut error_buf: ErrorBuffer,
) {
    let desc = desc.map_label(cow_label);
    let (_, error) = gfx_select!(self_id => global.device_create_buffer(self_id, &desc, new_id));
    if let Some(err) = error {
        error_buf.init(err);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_map(
    global: &Global,
    buffer_id: id::BufferId,
    start: wgt::BufferAddress,
    size: wgt::BufferAddress,
    operation: wgc::resource::BufferMapOperation,
) {
    gfx_select!(buffer_id => global.buffer_map_async(
        buffer_id,
        start .. start + size,
        operation
    ))
    .unwrap();
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `size` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_buffer_get_mapped_range(
    global: &Global,
    buffer_id: id::BufferId,
    start: wgt::BufferAddress,
    size: wgt::BufferAddress,
) -> *mut u8 {
    gfx_select!(buffer_id => global.buffer_get_mapped_range(
        buffer_id,
        start,
        Some(size)
    ))
    .unwrap()
    .0
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_unmap(global: &Global, buffer_id: id::BufferId) {
    gfx_select!(buffer_id => global.buffer_unmap(buffer_id)).unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_server_buffer_drop(global: &Global, self_id: id::BufferId) {
    gfx_select!(self_id => global.buffer_drop(self_id, false));
}

trait GlobalExt {
    fn device_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::DeviceId,
        action: DeviceAction,
        error_buf: ErrorBuffer,
    );
    fn texture_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::TextureId,
        action: TextureAction,
        error_buf: ErrorBuffer,
    );
    fn command_encoder_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::CommandEncoderId,
        action: CommandEncoderAction,
        error_buf: ErrorBuffer,
    );
}

impl GlobalExt for Global {
    fn device_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::DeviceId,
        action: DeviceAction,
        mut error_buf: ErrorBuffer,
    ) {
        match action {
            DeviceAction::CreateBuffer(id, desc) => {
                let (_, error) = self.device_create_buffer::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateTexture(id, desc) => {
                let (_, error) = self.device_create_texture::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateSampler(id, desc) => {
                let (_, error) = self.device_create_sampler::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateBindGroupLayout(id, desc) => {
                let (_, error) = self.device_create_bind_group_layout::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreatePipelineLayout(id, desc) => {
                let (_, error) = self.device_create_pipeline_layout::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateBindGroup(id, desc) => {
                let (_, error) = self.device_create_bind_group::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateShaderModule(id, desc, source) => {
                let source = match source {
                    ShaderModuleSource::SpirV(data) => {
                        wgc::pipeline::ShaderModuleSource::SpirV(data)
                    }
                    ShaderModuleSource::Wgsl(data) => wgc::pipeline::ShaderModuleSource::Wgsl(data),
                };
                let (_, error) = self.device_create_shader_module::<B>(self_id, &desc, source, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateComputePipeline(id, desc, implicit) => {
                let implicit_ids = implicit
                    .as_ref()
                    .map(|imp| wgc::device::ImplicitPipelineIds {
                        root_id: imp.pipeline,
                        group_ids: &imp.bind_groups,
                    });
                let (_, error) =
                    self.device_create_compute_pipeline::<B>(self_id, &desc, id, implicit_ids);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateRenderPipeline(id, desc, implicit) => {
                let implicit_ids = implicit
                    .as_ref()
                    .map(|imp| wgc::device::ImplicitPipelineIds {
                        root_id: imp.pipeline,
                        group_ids: &imp.bind_groups,
                    });
                let (_, error) =
                    self.device_create_render_pipeline::<B>(self_id, &desc, id, implicit_ids);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateRenderBundle(id, encoder, desc) => {
                let (_, error) = self.render_bundle_encoder_finish::<B>(encoder, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
            DeviceAction::CreateCommandEncoder(id, desc) => {
                let (_, error) = self.device_create_command_encoder::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
        }
    }

    fn texture_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::TextureId,
        action: TextureAction,
        mut error_buf: ErrorBuffer,
    ) {
        match action {
            TextureAction::CreateView(id, desc) => {
                let (_, error) = self.texture_create_view::<B>(self_id, &desc, id);
                if let Some(err) = error {
                    error_buf.init(err);
                }
            }
        }
    }

    fn command_encoder_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::CommandEncoderId,
        action: CommandEncoderAction,
        mut error_buf: ErrorBuffer,
    ) {
        match action {
            CommandEncoderAction::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => {
                if let Err(err) = self.command_encoder_copy_buffer_to_buffer::<B>(
                    self_id, src, src_offset, dst, dst_offset, size,
                ) {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::CopyBufferToTexture { src, dst, size } => {
                if let Err(err) =
                    self.command_encoder_copy_buffer_to_texture::<B>(self_id, &src, &dst, &size)
                {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::CopyTextureToBuffer { src, dst, size } => {
                if let Err(err) =
                    self.command_encoder_copy_texture_to_buffer::<B>(self_id, &src, &dst, &size)
                {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::CopyTextureToTexture { src, dst, size } => {
                if let Err(err) =
                    self.command_encoder_copy_texture_to_texture::<B>(self_id, &src, &dst, &size)
                {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::RunComputePass { base } => {
                if let Err(err) =
                    self.command_encoder_run_compute_pass_impl::<B>(self_id, base.as_ref())
                {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::WriteTimestamp {
                query_set_id,
                query_index,
            } => {
                if let Err(err) =
                    self.command_encoder_write_timestamp::<B>(self_id, query_set_id, query_index)
                {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::ResolveQuerySet {
                query_set_id,
                start_query,
                query_count,
                destination,
                destination_offset,
            } => {
                if let Err(err) = self.command_encoder_resolve_query_set::<B>(
                    self_id,
                    query_set_id,
                    start_query,
                    query_count,
                    destination,
                    destination_offset,
                ) {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::RunRenderPass {
                base,
                target_colors,
                target_depth_stencil,
            } => {
                if let Err(err) = self.command_encoder_run_render_pass_impl::<B>(
                    self_id,
                    base.as_ref(),
                    &target_colors,
                    target_depth_stencil.as_ref(),
                ) {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::ClearBuffer { dst, offset, size } => {
                if let Err(err) = self.command_encoder_clear_buffer::<B>(self_id, dst, offset, size)
                {
                    error_buf.init(err);
                }
            }
            CommandEncoderAction::ClearImage {
                dst,
                ref subresource_range,
            } => {
                if let Err(err) =
                    self.command_encoder_clear_image::<B>(self_id, dst, subresource_range)
                {
                    error_buf.init(err);
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_device_action(
    global: &Global,
    self_id: id::DeviceId,
    byte_buf: &ByteBuf,
    error_buf: ErrorBuffer,
) {
    let action = bincode::deserialize(byte_buf.as_slice()).unwrap();
    gfx_select!(self_id => global.device_action(self_id, action, error_buf));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_texture_action(
    global: &Global,
    self_id: id::TextureId,
    byte_buf: &ByteBuf,
    error_buf: ErrorBuffer,
) {
    let action = bincode::deserialize(byte_buf.as_slice()).unwrap();
    gfx_select!(self_id => global.texture_action(self_id, action, error_buf));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_command_encoder_action(
    global: &Global,
    self_id: id::CommandEncoderId,
    byte_buf: &ByteBuf,
    error_buf: ErrorBuffer,
) {
    let action = bincode::deserialize(byte_buf.as_slice()).unwrap();
    gfx_select!(self_id => global.command_encoder_action(self_id, action, error_buf));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_encoder(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::CommandEncoderDescriptor<RawString>,
    new_id: id::CommandEncoderId,
    mut error_buf: ErrorBuffer,
) {
    let desc = desc.map_label(cow_label);
    let (_, error) =
        gfx_select!(self_id => global.device_create_command_encoder(self_id, &desc, new_id));
    if let Some(err) = error {
        error_buf.init(err);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_finish(
    global: &Global,
    self_id: id::CommandEncoderId,
    desc: &wgt::CommandBufferDescriptor<RawString>,
    mut error_buf: ErrorBuffer,
) {
    let desc = desc.map_label(cow_label);
    let (_, error) = gfx_select!(self_id => global.command_encoder_finish(self_id, &desc));
    if let Some(err) = error {
        error_buf.init(err);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_drop(global: &Global, self_id: id::CommandEncoderId) {
    gfx_select!(self_id => global.command_encoder_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_command_buffer_drop(global: &Global, self_id: id::CommandBufferId) {
    gfx_select!(self_id => global.command_buffer_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_render_bundle_drop(global: &Global, self_id: id::RenderBundleId) {
    gfx_select!(self_id => global.render_bundle_drop(self_id));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_texture_to_buffer(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::ImageCopyTexture,
    destination: &wgc::command::ImageCopyBuffer,
    size: &wgt::Extent3d,
) {
    gfx_select!(self_id => global.command_encoder_copy_texture_to_buffer(self_id, source, destination, size)).unwrap();
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `command_buffer_id_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_submit(
    global: &Global,
    self_id: id::QueueId,
    command_buffer_ids: *const id::CommandBufferId,
    command_buffer_id_length: usize,
    mut error_buf: ErrorBuffer,
) {
    let command_buffers = slice::from_raw_parts(command_buffer_ids, command_buffer_id_length);
    let result = gfx_select!(self_id => global.queue_submit(self_id, command_buffers));
    if let Err(err) = result {
        error_buf.init(err);
    }
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `data_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_write_action(
    global: &Global,
    self_id: id::QueueId,
    byte_buf: &ByteBuf,
    data: *const u8,
    data_length: usize,
    mut error_buf: ErrorBuffer,
) {
    let action: QueueWriteAction = bincode::deserialize(byte_buf.as_slice()).unwrap();
    let data = slice::from_raw_parts(data, data_length);
    let result = match action {
        QueueWriteAction::Buffer { dst, offset } => {
            gfx_select!(self_id => global.queue_write_buffer(self_id, dst, offset, data))
        }
        QueueWriteAction::Texture { dst, layout, size } => {
            gfx_select!(self_id => global.queue_write_texture(self_id, &dst, &data, &layout, &size))
        }
    };
    if let Err(err) = result {
        error_buf.init(err);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_layout_drop(
    global: &Global,
    self_id: id::BindGroupLayoutId,
) {
    gfx_select!(self_id => global.bind_group_layout_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_pipeline_layout_drop(global: &Global, self_id: id::PipelineLayoutId) {
    gfx_select!(self_id => global.pipeline_layout_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_drop(global: &Global, self_id: id::BindGroupId) {
    gfx_select!(self_id => global.bind_group_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_shader_module_drop(global: &Global, self_id: id::ShaderModuleId) {
    gfx_select!(self_id => global.shader_module_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_compute_pipeline_drop(
    global: &Global,
    self_id: id::ComputePipelineId,
) {
    gfx_select!(self_id => global.compute_pipeline_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_render_pipeline_drop(global: &Global, self_id: id::RenderPipelineId) {
    gfx_select!(self_id => global.render_pipeline_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_texture_drop(global: &Global, self_id: id::TextureId) {
    gfx_select!(self_id => global.texture_drop(self_id, false));
}

#[no_mangle]
pub extern "C" fn wgpu_server_texture_view_drop(global: &Global, self_id: id::TextureViewId) {
    gfx_select!(self_id => global.texture_view_drop(self_id, false)).unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_server_sampler_drop(global: &Global, self_id: id::SamplerId) {
    gfx_select!(self_id => global.sampler_drop(self_id));
}

#[no_mangle]
pub extern "C" fn wgpu_server_compute_pipeline_get_bind_group_layout(
    global: &Global,
    self_id: id::ComputePipelineId,
    index: u32,
    assign_id: id::BindGroupLayoutId,
    mut error_buf: ErrorBuffer,
) {
    let (_, error) = gfx_select!(self_id => global.compute_pipeline_get_bind_group_layout(self_id, index, assign_id));
    if let Some(err) = error {
        error_buf.init(err);
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_render_pipeline_get_bind_group_layout(
    global: &Global,
    self_id: id::RenderPipelineId,
    index: u32,
    assign_id: id::BindGroupLayoutId,
    mut error_buf: ErrorBuffer,
) {
    let (_, error) = gfx_select!(self_id => global.render_pipeline_get_bind_group_layout(self_id, index, assign_id));
    if let Some(err) = error {
        error_buf.init(err);
    }
}

/// Encode the freeing of the selected ID into a byte buf.
#[no_mangle]
pub extern "C" fn wgpu_server_adapter_free(id: id::AdapterId, drop_byte_buf: &mut ByteBuf) {
    *drop_byte_buf = DropAction::Adapter(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_device_free(id: id::DeviceId, drop_byte_buf: &mut ByteBuf) {
    *drop_byte_buf = DropAction::Device(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_shader_module_free(
    id: id::ShaderModuleId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::ShaderModule(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_pipeline_layout_free(
    id: id::PipelineLayoutId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::PipelineLayout(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_layout_free(
    id: id::BindGroupLayoutId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::BindGroupLayout(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_bind_group_free(id: id::BindGroupId, drop_byte_buf: &mut ByteBuf) {
    *drop_byte_buf = DropAction::BindGroup(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_command_buffer_free(
    id: id::CommandBufferId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::CommandBuffer(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_render_bundle_free(
    id: id::RenderBundleId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::RenderBundle(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_render_pipeline_free(
    id: id::RenderPipelineId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::RenderPipeline(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_compute_pipeline_free(
    id: id::ComputePipelineId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::ComputePipeline(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_buffer_free(id: id::BufferId, drop_byte_buf: &mut ByteBuf) {
    *drop_byte_buf = DropAction::Buffer(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_texture_free(id: id::TextureId, drop_byte_buf: &mut ByteBuf) {
    *drop_byte_buf = DropAction::Texture(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_texture_view_free(
    id: id::TextureViewId,
    drop_byte_buf: &mut ByteBuf,
) {
    *drop_byte_buf = DropAction::TextureView(id).to_byte_buf();
}
#[no_mangle]
pub extern "C" fn wgpu_server_sampler_free(id: id::SamplerId, drop_byte_buf: &mut ByteBuf) {
    *drop_byte_buf = DropAction::Sampler(id).to_byte_buf();
}
