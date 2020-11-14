/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    cow_label, identity::IdentityRecyclerFactory, ByteBuf, CommandEncoderAction, DeviceAction,
    DropAction, RawString, TextureAction,
};

use wgc::{gfx_select, id};

use std::slice;

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
) -> i8 {
    let ids = slice::from_raw_parts(ids, id_length);
    match global.request_adapter(
        desc,
        wgc::instance::AdapterInputs::IdSet(ids, |i| i.backend()),
    ) {
        Ok(id) => ids.iter().position(|&i| i == id).unwrap() as i8,
        Err(e) => {
            log::warn!("request_adapter: {:?}", e);
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_server_fill_default_limits(limits: &mut wgt::Limits) {
    *limits = wgt::Limits::default();
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_adapter_request_device(
    global: &Global,
    self_id: id::AdapterId,
    desc: &wgt::DeviceDescriptor,
    new_id: id::DeviceId,
) {
    let trace_string = std::env::var("WGPU_TRACE").ok();
    let trace_path = trace_string
        .as_ref()
        .map(|string| std::path::Path::new(string.as_str()));
    gfx_select!(self_id => global.adapter_request_device(self_id, desc, trace_path, new_id))
        .unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_server_adapter_drop(global: &Global, adapter_id: id::AdapterId) {
    gfx_select!(adapter_id => global.adapter_destroy(adapter_id))
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
) {
    let desc = desc.map_label(cow_label);
    gfx_select!(self_id => global.device_create_buffer(self_id, &desc, new_id)).unwrap();
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
    size: Option<wgt::BufferSize>,
) -> *mut u8 {
    gfx_select!(buffer_id => global.buffer_get_mapped_range(
        buffer_id,
        start,
        size
    ))
    .unwrap()
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
    ) -> Vec<u8>;
    fn texture_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::TextureId,
        action: TextureAction,
    );
    fn command_encoder_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::CommandEncoderId,
        action: CommandEncoderAction,
    );
}

impl GlobalExt for Global {
    fn device_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::DeviceId,
        action: DeviceAction,
    ) -> Vec<u8> {
        let mut drop_actions = Vec::new();
        match action {
            DeviceAction::CreateBuffer(id, desc) => {
                self.device_create_buffer::<B>(self_id, &desc, id).unwrap();
            }
            DeviceAction::CreateTexture(id, desc) => {
                self.device_create_texture::<B>(self_id, &desc, id).unwrap();
            }
            DeviceAction::CreateSampler(id, desc) => {
                self.device_create_sampler::<B>(self_id, &desc, id).unwrap();
            }
            DeviceAction::CreateBindGroupLayout(id, desc) => {
                self.device_create_bind_group_layout::<B>(self_id, &desc, id)
                    .unwrap();
            }
            DeviceAction::CreatePipelineLayout(id, desc) => {
                self.device_create_pipeline_layout::<B>(self_id, &desc, id)
                    .unwrap();
            }
            DeviceAction::CreateBindGroup(id, desc) => {
                self.device_create_bind_group::<B>(self_id, &desc, id)
                    .unwrap();
            }
            DeviceAction::CreateShaderModule(id, spirv, wgsl) => {
                let source = if spirv.is_empty() {
                    wgc::pipeline::ShaderModuleSource::Wgsl(wgsl)
                } else {
                    wgc::pipeline::ShaderModuleSource::SpirV(spirv)
                };
                self.device_create_shader_module::<B>(self_id, source, id)
                    .unwrap();
            }
            DeviceAction::CreateComputePipeline(id, desc, implicit) => {
                let implicit_ids = implicit
                    .as_ref()
                    .map(|imp| wgc::device::ImplicitPipelineIds {
                        root_id: imp.pipeline,
                        group_ids: &imp.bind_groups,
                    });
                let (_, group_count) = self
                    .device_create_compute_pipeline::<B>(self_id, &desc, id, implicit_ids)
                    .unwrap();
                if let Some(ref imp) = implicit {
                    for &bgl_id in imp.bind_groups[group_count as usize..].iter() {
                        bincode::serialize_into(
                            &mut drop_actions,
                            &DropAction::BindGroupLayout(bgl_id),
                        )
                        .unwrap();
                    }
                }
            }
            DeviceAction::CreateRenderPipeline(id, desc, implicit) => {
                let implicit_ids = implicit
                    .as_ref()
                    .map(|imp| wgc::device::ImplicitPipelineIds {
                        root_id: imp.pipeline,
                        group_ids: &imp.bind_groups,
                    });
                let (_, group_count) = self
                    .device_create_render_pipeline::<B>(self_id, &desc, id, implicit_ids)
                    .unwrap();
                if let Some(ref imp) = implicit {
                    for &bgl_id in imp.bind_groups[group_count as usize..].iter() {
                        bincode::serialize_into(
                            &mut drop_actions,
                            &DropAction::BindGroupLayout(bgl_id),
                        )
                        .unwrap();
                    }
                }
            }
            DeviceAction::CreateRenderBundle(_id, desc, _base) => {
                wgc::command::RenderBundleEncoder::new(&desc, self_id, None).unwrap();
            }
            DeviceAction::CreateCommandEncoder(id, desc) => {
                self.device_create_command_encoder::<B>(self_id, &desc, id)
                    .unwrap();
            }
        }
        drop_actions
    }

    fn texture_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::TextureId,
        action: TextureAction,
    ) {
        match action {
            TextureAction::CreateView(id, desc) => {
                self.texture_create_view::<B>(self_id, &desc, id).unwrap();
            }
        }
    }

    fn command_encoder_action<B: wgc::hub::GfxBackend>(
        &self,
        self_id: id::CommandEncoderId,
        action: CommandEncoderAction,
    ) {
        match action {
            CommandEncoderAction::CopyBufferToBuffer {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            } => self
                .command_encoder_copy_buffer_to_buffer::<B>(
                    self_id, src, src_offset, dst, dst_offset, size,
                )
                .unwrap(),
            CommandEncoderAction::CopyBufferToTexture { src, dst, size } => self
                .command_encoder_copy_buffer_to_texture::<B>(self_id, &src, &dst, &size)
                .unwrap(),
            CommandEncoderAction::CopyTextureToBuffer { src, dst, size } => self
                .command_encoder_copy_texture_to_buffer::<B>(self_id, &src, &dst, &size)
                .unwrap(),
            CommandEncoderAction::CopyTextureToTexture { src, dst, size } => self
                .command_encoder_copy_texture_to_texture::<B>(self_id, &src, &dst, &size)
                .unwrap(),
            CommandEncoderAction::RunComputePass { base } => {
                self.command_encoder_run_compute_pass_impl::<B>(self_id, base.as_ref())
                    .unwrap();
            }
            CommandEncoderAction::RunRenderPass {
                base,
                target_colors,
                target_depth_stencil,
            } => {
                self.command_encoder_run_render_pass_impl::<B>(
                    self_id,
                    base.as_ref(),
                    &target_colors,
                    target_depth_stencil.as_ref(),
                )
                .unwrap();
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_device_action(
    global: &Global,
    self_id: id::DeviceId,
    byte_buf: &ByteBuf,
    drop_byte_buf: &mut ByteBuf,
) {
    let action = bincode::deserialize(byte_buf.as_slice()).unwrap();
    let drop_actions = gfx_select!(self_id => global.device_action(self_id, action));
    *drop_byte_buf = ByteBuf::from_vec(drop_actions);
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_texture_action(
    global: &Global,
    self_id: id::TextureId,
    byte_buf: &ByteBuf,
) {
    let action = bincode::deserialize(byte_buf.as_slice()).unwrap();
    gfx_select!(self_id => global.texture_action(self_id, action));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_command_encoder_action(
    global: &Global,
    self_id: id::CommandEncoderId,
    byte_buf: &ByteBuf,
) {
    let action = bincode::deserialize(byte_buf.as_slice()).unwrap();
    gfx_select!(self_id => global.command_encoder_action(self_id, action));
}

#[no_mangle]
pub extern "C" fn wgpu_server_device_create_encoder(
    global: &Global,
    self_id: id::DeviceId,
    desc: &wgt::CommandEncoderDescriptor<RawString>,
    new_id: id::CommandEncoderId,
) {
    let desc = desc.map_label(cow_label);
    gfx_select!(self_id => global.device_create_command_encoder(self_id, &desc, new_id)).unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_finish(
    global: &Global,
    self_id: id::CommandEncoderId,
    desc: &wgt::CommandBufferDescriptor<RawString>,
) {
    let desc = desc.map_label(cow_label);
    gfx_select!(self_id => global.command_encoder_finish(self_id, &desc)).unwrap();
}

#[no_mangle]
pub extern "C" fn wgpu_server_encoder_drop(global: &Global, self_id: id::CommandEncoderId) {
    gfx_select!(self_id => global.command_encoder_drop(self_id));
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `byte_length` elements.
#[no_mangle]
pub extern "C" fn wgpu_server_command_buffer_drop(global: &Global, self_id: id::CommandBufferId) {
    gfx_select!(self_id => global.command_buffer_drop(self_id));
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_server_encoder_copy_texture_to_buffer(
    global: &Global,
    self_id: id::CommandEncoderId,
    source: &wgc::command::TextureCopyView,
    destination: &wgc::command::BufferCopyView,
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
) {
    let command_buffers = slice::from_raw_parts(command_buffer_ids, command_buffer_id_length);
    gfx_select!(self_id => global.queue_submit(self_id, command_buffers)).unwrap();
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `data_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_write_buffer(
    global: &Global,
    self_id: id::QueueId,
    buffer_id: id::BufferId,
    buffer_offset: wgt::BufferAddress,
    data: *const u8,
    data_length: usize,
) {
    let data = slice::from_raw_parts(data, data_length);
    gfx_select!(self_id => global.queue_write_buffer(self_id, buffer_id, buffer_offset, data))
        .unwrap();
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `data_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_server_queue_write_texture(
    global: &Global,
    self_id: id::QueueId,
    destination: &wgc::command::TextureCopyView,
    data: *const u8,
    data_length: usize,
    layout: &wgt::TextureDataLayout,
    extent: &wgt::Extent3d,
) {
    let data = slice::from_raw_parts(data, data_length);
    gfx_select!(self_id => global.queue_write_texture(self_id, destination, data, layout, extent))
        .unwrap();
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
    gfx_select!(self_id => global.texture_view_drop(self_id)).unwrap();
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
) -> id::BindGroupLayoutId {
    gfx_select!(self_id => global.compute_pipeline_get_bind_group_layout(self_id, index)).unwrap()
}

#[no_mangle]
pub extern "C" fn wgpu_server_render_pipeline_get_bind_group_layout(
    global: &Global,
    self_id: id::RenderPipelineId,
    index: u32,
) -> id::BindGroupLayoutId {
    gfx_select!(self_id => global.render_pipeline_get_bind_group_layout(self_id, index)).unwrap()
}
