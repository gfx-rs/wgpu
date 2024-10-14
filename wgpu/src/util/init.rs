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

/// Determines whether the [`Backends::BROWSER_WEBGPU`] backend is supported.
///
/// The result can only be true if this is called from the main thread or a dedicated worker.
/// For convenience, this is also supported on non-wasm targets, always returning false there.
pub async fn is_browser_webgpu_supported() -> bool {
    #[cfg(webgpu)]
    {
        // In theory it should be enough to check for the presence of the `gpu` property...
        let gpu = crate::backend::get_browser_gpu_property();
        let Ok(Some(gpu)) = gpu else {
            return false;
        };

        // ...but in practice, we also have to try to create an adapter, since as of writing
        // Chrome on Linux has the `gpu` property but doesn't support WebGPU.
        let adapter_promise = gpu.request_adapter();
        wasm_bindgen_futures::JsFuture::from(adapter_promise)
            .await
            .map_or(false, |adapter| {
                !adapter.is_undefined() && !adapter.is_null()
            })
    }
    #[cfg(not(webgpu))]
    {
        false
    }
}

/// Create an new instance of wgpu, but disabling [`Backends::BROWSER_WEBGPU`] if no WebGPU support was detected.
///
/// If the instance descriptor enables [`Backends::BROWSER_WEBGPU`],
/// this checks via [`is_browser_webgpu_supported`] for WebGPU support before forwarding
/// the descriptor with or without [`Backends::BROWSER_WEBGPU`] respecitively to [`Instance::new`].
///
/// You should prefer this method over [`Instance::new`] if you want to target WebGPU and automatically
/// fall back to WebGL if WebGPU is not available.
/// This is because WebGPU support has to be decided upon instance creation and [`Instance::new`]
/// (being a `sync` function) can't establish WebGPU support (details see [`is_browser_webgpu_supported`]).
///
/// # Panics
///
/// If no backend feature for the active target platform is enabled,
/// this method will panic, see [`Instance::enabled_backend_features()`].
#[allow(unused_mut)]
pub async fn new_instance_with_webgpu_detection(
    mut instance_desc: wgt::InstanceDescriptor,
) -> crate::Instance {
    if instance_desc
        .backends
        .contains(wgt::Backends::BROWSER_WEBGPU)
        && !is_browser_webgpu_supported().await
    {
        instance_desc.backends.remove(wgt::Backends::BROWSER_WEBGPU);
    }

    crate::Instance::new(instance_desc)
}
