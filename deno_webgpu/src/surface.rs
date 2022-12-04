use deno_core::error::AnyError;
use deno_core::include_js_files;
use deno_core::Extension;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use serde::Deserialize;
use std::borrow::Cow;

pub fn init_surface(unstable: bool) -> Extension {
    Extension::builder()
        .js(include_js_files!(
          prefix "deno:deno_webgpu",
          "03_surface.js",
          "04_surface_idl_types.js",
        ))
        .ops(declare_webgpu_surface_ops())
        .state(move |state| {
            // TODO: check & possibly streamline this
            // Unstable might be able to be OpMiddleware
            // let unstable_checker = state.borrow::<super::UnstableChecker>();
            // let unstable = unstable_checker.unstable;
            state.put(super::Unstable(unstable));
            Ok(())
        })
        .build()
}

struct WebGpuSurface(wgpu_core::id::SurfaceId);
impl Resource for WebGpuSurface {
    fn name(&self) -> Cow<str> {
        "webGPUSurface".into()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceConfigureArgs {
    surface_rid: ResourceId,
    device_rid: ResourceId,
    format: wgpu_types::TextureFormat,
    usage: u32,
    width: u32,
    height: u32,
    alpha_mode: wgpu_types::CompositeAlphaMode,
}

#[op]
pub fn op_webgpu_surface_configure(
    state: &mut OpState,
    args: SurfaceConfigureArgs,
) -> Result<super::WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let device_resource = state
        .resource_table
        .get::<super::WebGpuDevice>(args.device_rid)?;
    let device = device_resource.0;
    let surface_resource = state
        .resource_table
        .get::<WebGpuSurface>(args.surface_rid)?;
    let surface = surface_resource.0;

    let conf = wgpu_types::SurfaceConfiguration {
        usage: wgpu_types::TextureUsages::from_bits_truncate(args.usage),
        format: args.format,
        width: args.width,
        height: args.height,
        present_mode: Default::default(), // TODO
        alpha_mode: args.alpha_mode,
    };

    let err = gfx_select!(surface => instance.surface_configure(surface, device, &conf));

    Ok(crate::error::WebGpuResult::maybe_err(err))
}

#[op]
pub fn op_webgpu_surface_get_current_texture(
    state: &mut OpState,
    surface_rid: ResourceId,
) -> Result<super::WebGpuResult, AnyError> {
    let instance = state.borrow::<super::Instance>();
    let surface_resource = state.resource_table.get::<WebGpuSurface>(surface_rid)?;
    let surface = surface_resource.0;

    let x = gfx_select!(surface => instance.surface_get_current_texture(surface, ()));

    Ok()
}
