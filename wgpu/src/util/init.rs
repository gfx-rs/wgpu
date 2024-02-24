use wgt::{Backends, PowerPreference, RequestAdapterOptions};

use crate::{Adapter, Instance, Surface};

#[cfg(wgpu_core)]
#[cfg_attr(docsrs, doc(cfg(all())))]
pub use wgc::instance::parse_backends_from_comma_list;
/// Just return ALL, if wgpu_core is not enabled.
#[cfg(not(wgpu_core))]
pub fn parse_backends_from_comma_list(_string: &str) -> Backends {
    Backends::all()
}

/// Get a set of backend bits from the environment variable WGPU_BACKEND.
pub fn backend_bits_from_env() -> Option<Backends> {
    std::env::var("WGPU_BACKEND")
        .as_deref()
        .map(str::to_lowercase)
        .ok()
        .as_deref()
        .map(parse_backends_from_comma_list)
}

/// Get a power preference from the environment variable WGPU_POWER_PREF
pub fn power_preference_from_env() -> Option<PowerPreference> {
    Some(
        match std::env::var("WGPU_POWER_PREF")
            .as_deref()
            .map(str::to_lowercase)
            .as_deref()
        {
            Ok("low") => PowerPreference::LowPower,
            Ok("high") => PowerPreference::HighPerformance,
            Ok("none") => PowerPreference::None,
            _ => return None,
        },
    )
}

/// Initialize the adapter obeying the WGPU_ADAPTER_NAME environment variable.
#[cfg(native)]
pub fn initialize_adapter_from_env(
    instance: &Instance,
    compatible_surface: Option<&Surface<'_>>,
) -> Option<Adapter> {
    let desired_adapter_name = std::env::var("WGPU_ADAPTER_NAME")
        .as_deref()
        .map(str::to_lowercase)
        .ok()?;

    let adapters = instance.enumerate_adapters(Backends::all());

    let mut chosen_adapter = None;
    for adapter in adapters {
        let info = adapter.get_info();

        if let Some(surface) = compatible_surface {
            if !adapter.is_surface_supported(surface) {
                continue;
            }
        }

        if info.name.to_lowercase().contains(&desired_adapter_name) {
            chosen_adapter = Some(adapter);
            break;
        }
    }

    Some(chosen_adapter.expect("WGPU_ADAPTER_NAME set but no matching adapter found!"))
}

/// Initialize the adapter obeying the WGPU_ADAPTER_NAME environment variable.
#[cfg(not(native))]
pub fn initialize_adapter_from_env(
    _instance: &Instance,
    _compatible_surface: Option<&Surface<'_>>,
) -> Option<Adapter> {
    None
}

/// Initialize the adapter obeying the WGPU_ADAPTER_NAME environment variable and if it doesn't exist fall back on a default adapter.
pub async fn initialize_adapter_from_env_or_default(
    instance: &Instance,
    compatible_surface: Option<&Surface<'_>>,
) -> Option<Adapter> {
    match initialize_adapter_from_env(instance, compatible_surface) {
        Some(a) => Some(a),
        None => {
            instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: power_preference_from_env().unwrap_or_default(),
                    force_fallback_adapter: false,
                    compatible_surface,
                })
                .await
        }
    }
}

/// Choose which DX12 shader compiler to use from the environment variable `WGPU_DX12_COMPILER`.
///
/// Possible values are `dxc` and `fxc`. Case insensitive.
pub fn dx12_shader_compiler_from_env() -> Option<wgt::Dx12Compiler> {
    Some(
        match std::env::var("WGPU_DX12_COMPILER")
            .as_deref()
            .map(str::to_lowercase)
            .as_deref()
        {
            Ok("dxc") => wgt::Dx12Compiler::Dxc {
                dxil_path: None,
                dxc_path: None,
            },
            Ok("fxc") => wgt::Dx12Compiler::Fxc,
            _ => return None,
        },
    )
}

/// Choose which minor OpenGL ES version to use from the environment variable `WGPU_GLES_MINOR_VERSION`.
///
/// Possible values are `0`, `1`, `2` or `automatic`. Case insensitive.
pub fn gles_minor_version_from_env() -> Option<wgt::Gles3MinorVersion> {
    Some(
        match std::env::var("WGPU_GLES_MINOR_VERSION")
            .as_deref()
            .map(str::to_lowercase)
            .as_deref()
        {
            Ok("automatic") => wgt::Gles3MinorVersion::Automatic,
            Ok("0") => wgt::Gles3MinorVersion::Version0,
            Ok("1") => wgt::Gles3MinorVersion::Version1,
            Ok("2") => wgt::Gles3MinorVersion::Version2,
            _ => return None,
        },
    )
}
