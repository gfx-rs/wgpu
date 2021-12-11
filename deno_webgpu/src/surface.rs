use deno_core::error::anyhow;
use deno_core::error::AnyError;
use deno_core::op_sync;
use deno_core::Extension;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use raw_window_handle::HasRawWindowHandle;
use raw_window_handle::RawWindowHandle;
use serde::Deserialize;
use std::borrow::Cow;
use wgpu_types::TextureFormat;

use crate::texture::WebGpuTexture;
use crate::Instance;
use crate::WebGpuAdapter;
use crate::WebGpuDevice;

pub struct DynHasRawWindowHandle(Box<dyn HasRawWindowHandle>);

unsafe impl HasRawWindowHandle for DynHasRawWindowHandle {
    fn raw_window_handle(&self) -> RawWindowHandle {
        self.0.raw_window_handle()
    }
}

pub struct WindowResource(DynHasRawWindowHandle);
impl Resource for WindowResource {
    fn name(&self) -> Cow<str> {
        "window".into()
    }
}

pub struct WebGpuSurface(wgpu_core::id::SurfaceId);
impl Resource for WebGpuSurface {
    fn name(&self) -> Cow<str> {
        "webGPUSurface".into()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSurfaceArgs {
    window_rid: ResourceId,
}

pub fn op_webgpu_create_surface(
    state: &mut OpState,
    args: CreateSurfaceArgs,
    _: (),
) -> Result<ResourceId, AnyError> {
    let window = state
        .resource_table
        .get::<WindowResource>(args.window_rid)?;
    let instance = state.borrow::<Instance>();
    let surface = instance.instance_create_surface(&window.0, std::marker::PhantomData);
    Ok(state.resource_table.add(WebGpuSurface(surface)))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigureSurfaceArgs {
    device_rid: ResourceId,
    surface_rid: ResourceId,
    format: TextureFormat,
    usage: u32,
    width: u32,
    height: u32,
}

pub fn op_webgpu_configure_surface(
    state: &mut OpState,
    args: ConfigureSurfaceArgs,
    _: (),
) -> Result<(), AnyError> {
    let surface = state
        .resource_table
        .get::<WebGpuSurface>(args.surface_rid)?;
    let device = state.resource_table.get::<WebGpuDevice>(args.device_rid)?;
    let instance = state.borrow::<Instance>();

    let config = wgpu_types::SurfaceConfiguration {
        usage: wgpu_types::TextureUsages::from_bits(args.usage).unwrap(),
        format: args.format,
        width: args.width,
        height: args.height,
        present_mode: wgpu_types::PresentMode::Fifo,
    };

    match gfx_select!(device.0 => instance.surface_configure(
        surface.0,
        device.0,
        &config
    )) {
        Some(err) => Err(err.into()),
        None => Ok(()),
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfacePreferredFormatArgs {
    adapter_rid: ResourceId,
    surface_rid: ResourceId,
}

pub fn op_webgpu_surface_get_preferred_format(
    state: &mut OpState,
    args: SurfacePreferredFormatArgs,
    _: (),
) -> Result<wgpu_types::TextureFormat, AnyError> {
    let surface = state
        .resource_table
        .get::<WebGpuSurface>(args.surface_rid)?;
    let adapter = state
        .resource_table
        .get::<WebGpuAdapter>(args.adapter_rid)?;
    let instance = state.borrow::<Instance>();

    match gfx_select!(adapter.0 => instance.surface_get_preferred_format(
        surface.0,
        adapter.0
    )) {
        Ok(format) => Ok(format),
        Err(err) => Err(err.into()),
    }
}

pub fn op_webgpu_surface_get_current_texture(
    state: &mut OpState,
    args: SurfacePreferredFormatArgs,
    _: (),
) -> Result<ResourceId, AnyError> {
    let surface = state
        .resource_table
        .get::<WebGpuSurface>(args.surface_rid)?;
    let adapter = state
        .resource_table
        .get::<WebGpuAdapter>(args.adapter_rid)?;
    let instance = state.borrow::<Instance>();

    let output = gfx_select!(
        adapter.0 => instance.surface_get_current_texture(surface.0, std::marker::PhantomData)
    )?;

    if let Some(texture) = output.texture_id {
        Ok(state.resource_table.add(WebGpuTexture(texture)))
    } else {
        Err(anyhow!(
            "Failed to get current texture. Surface Status: {:?}",
            output.status
        ))
    }
}

pub fn op_webgpu_surface_present(
    state: &mut OpState,
    args: SurfacePreferredFormatArgs,
    _: (),
) -> Result<String, AnyError> {
    let surface = state
        .resource_table
        .get::<WebGpuSurface>(args.surface_rid)?;
    let adapter = state
        .resource_table
        .get::<WebGpuAdapter>(args.adapter_rid)?;
    let instance = state.borrow::<Instance>();

    let status = gfx_select!(adapter.0 => instance.surface_present(surface.0))?;

    Ok(String::from(match status {
        wgpu_types::SurfaceStatus::Good => "good",
        wgpu_types::SurfaceStatus::Suboptimal => "suboptimal",
        wgpu_types::SurfaceStatus::Timeout => "timeout",
        wgpu_types::SurfaceStatus::Outdated => "outdated",
        wgpu_types::SurfaceStatus::Lost => "lost",
    }))
}

pub fn op_webgpu_surface_drop(state: &mut OpState, rid: ResourceId, _: ()) -> Result<(), AnyError> {
    let surface = state.resource_table.get::<WebGpuSurface>(rid)?;
    let instance = state.borrow::<Instance>();
    instance.surface_drop(surface.0);
    Ok(())
}

pub fn init_surface_ext() -> Extension {
    Extension::builder()
        .ops(vec![
            (
                "op_webgpu_create_surface",
                op_sync(op_webgpu_create_surface),
            ),
            (
                "op_webgpu_configure_surface",
                op_sync(op_webgpu_configure_surface),
            ),
            (
                "op_webgpu_surface_get_preferred_format",
                op_sync(op_webgpu_surface_get_preferred_format),
            ),
            (
                "op_webgpu_surface_get_current_texture",
                op_sync(op_webgpu_surface_get_current_texture),
            ),
            (
                "op_webgpu_surface_present",
                op_sync(op_webgpu_surface_present),
            ),
            ("op_webgpu_surface_drop", op_sync(op_webgpu_surface_drop)),
        ])
        .build()
}
