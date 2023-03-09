// Copyright 2018-2023 the Deno authors. All rights reserved. MIT license.

use deno_core::error::AnyError;
use deno_core::op;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use serde::Deserialize;
use std::borrow::Cow;

use super::error::WebGpuResult;
pub(crate) struct WebGpuTexture(pub(crate) wgpu_core::id::TextureId);
impl Resource for WebGpuTexture {
    fn name(&self) -> Cow<str> {
        "webGPUTexture".into()
    }
}

pub(crate) struct WebGpuTextureView(pub(crate) wgpu_core::id::TextureViewId);
impl Resource for WebGpuTextureView {
    fn name(&self) -> Cow<str> {
        "webGPUTextureView".into()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateTextureArgs {
    device_rid: ResourceId,
    label: Option<String>,
    size: wgpu_types::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
    dimension: wgpu_types::TextureDimension,
    format: wgpu_types::TextureFormat,
    usage: u32,
    view_formats: Vec<wgpu_types::TextureFormat>,
}

#[op]
pub fn op_webgpu_create_texture(
    state: &mut OpState,
    args: CreateTextureArgs,
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let device_resource = state
        .resource_table
        .get::<super::WebGpuDevice>(args.device_rid)?;
    let device = device_resource.0;

    let descriptor = wgpu_core::resource::TextureDescriptor {
        label: args.label.map(Cow::from),
        size: args.size,
        mip_level_count: args.mip_level_count,
        sample_count: args.sample_count,
        dimension: args.dimension,
        format: args.format,
        usage: wgpu_types::TextureUsages::from_bits_truncate(args.usage),
        view_formats: args.view_formats,
    };

    gfx_put!(device => instance.device_create_texture(
    device,
    &descriptor,
    ()
  ) => state, WebGpuTexture)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateTextureViewArgs {
    texture_rid: ResourceId,
    label: Option<String>,
    format: Option<wgpu_types::TextureFormat>,
    dimension: Option<wgpu_types::TextureViewDimension>,
    #[serde(flatten)]
    range: wgpu_types::ImageSubresourceRange,
}

#[op]
pub fn op_webgpu_create_texture_view(
    state: &mut OpState,
    args: CreateTextureViewArgs,
) -> Result<WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let texture_resource = state
        .resource_table
        .get::<WebGpuTexture>(args.texture_rid)?;
    let texture = texture_resource.0;

    let descriptor = wgpu_core::resource::TextureViewDescriptor {
        label: args.label.map(Cow::from),
        format: args.format,
        dimension: args.dimension,
        range: args.range,
    };

    gfx_put!(texture => instance.texture_create_view(
    texture,
    &descriptor,
    ()
  ) => state, WebGpuTextureView)
}
