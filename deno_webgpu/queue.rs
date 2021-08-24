// Copyright 2018-2021 the Deno authors. All rights reserved. MIT license.

use std::num::NonZeroU32;

use deno_core::error::null_opbuf;
use deno_core::error::AnyError;
use deno_core::OpState;
use deno_core::ResourceId;
use deno_core::ZeroCopyBuf;
use serde::Deserialize;

use super::error::WebGpuResult;

type WebGpuQueue = super::WebGpuDevice;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueSubmitArgs {
    queue_rid: ResourceId,
    command_buffers: Vec<ResourceId>,
}

pub fn op_webgpu_queue_submit(
    state: &mut OpState,
    args: QueueSubmitArgs,
    _: (),
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let queue_resource = state.resource_table.get::<WebGpuQueue>(args.queue_rid)?;
    let queue = queue_resource.0;

    let mut ids = vec![];

    for rid in args.command_buffers {
        let buffer_resource = state
            .resource_table
            .get::<super::command_encoder::WebGpuCommandBuffer>(rid)?;
        ids.push(buffer_resource.0);
    }

    let maybe_err = gfx_select!(queue => instance.queue_submit(queue, &ids)).err();

    Ok(WebGpuResult::maybe_err(maybe_err))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuImageDataLayout {
    offset: u64,
    bytes_per_row: Option<u32>,
    rows_per_image: Option<u32>,
}

impl From<GpuImageDataLayout> for wgpu_types::ImageDataLayout {
    fn from(layout: GpuImageDataLayout) -> Self {
        wgpu_types::ImageDataLayout {
            offset: layout.offset,
            bytes_per_row: NonZeroU32::new(layout.bytes_per_row.unwrap_or(0)),
            rows_per_image: NonZeroU32::new(layout.rows_per_image.unwrap_or(0)),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueWriteBufferArgs {
    queue_rid: ResourceId,
    buffer: ResourceId,
    buffer_offset: u64,
    data_offset: usize,
    size: Option<usize>,
}

pub fn op_webgpu_write_buffer(
    state: &mut OpState,
    args: QueueWriteBufferArgs,
    zero_copy: Option<ZeroCopyBuf>,
) -> Result<WebGpuResult, AnyError> {
    let zero_copy = zero_copy.ok_or_else(null_opbuf)?;
    let instance = state.borrow::<super::Instance>();
    let buffer_resource = state
        .resource_table
        .get::<super::buffer::WebGpuBuffer>(args.buffer)?;
    let buffer = buffer_resource.0;
    let queue_resource = state.resource_table.get::<WebGpuQueue>(args.queue_rid)?;
    let queue = queue_resource.0;

    let data = match args.size {
        Some(size) => &zero_copy[args.data_offset..(args.data_offset + size)],
        None => &zero_copy[args.data_offset..],
    };
    let maybe_err = gfx_select!(queue => instance.queue_write_buffer(
      queue,
      buffer,
      args.buffer_offset,
      data
    ))
    .err();

    Ok(WebGpuResult::maybe_err(maybe_err))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueWriteTextureArgs {
    queue_rid: ResourceId,
    destination: super::command_encoder::GpuImageCopyTexture,
    data_layout: GpuImageDataLayout,
    size: super::texture::GpuExtent3D,
}

pub fn op_webgpu_write_texture(
    state: &mut OpState,
    args: QueueWriteTextureArgs,
    zero_copy: Option<ZeroCopyBuf>,
) -> Result<WebGpuResult, AnyError> {
    let zero_copy = zero_copy.ok_or_else(null_opbuf)?;
    let instance = state.borrow::<super::Instance>();
    let texture_resource = state
        .resource_table
        .get::<super::texture::WebGpuTexture>(args.destination.texture)?;
    let queue_resource = state.resource_table.get::<WebGpuQueue>(args.queue_rid)?;
    let queue = queue_resource.0;

    let destination = wgpu_core::command::ImageCopyTexture {
        texture: texture_resource.0,
        mip_level: args.destination.mip_level,
        origin: args.destination.origin.into(),
        aspect: args.destination.aspect.into(),
    };
    let data_layout = args.data_layout.into();

    gfx_ok!(queue => instance.queue_write_texture(
      queue,
      &destination,
      &*zero_copy,
      &data_layout,
      &args.size.into()
    ))
}
