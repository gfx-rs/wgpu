use wgt::{Backends, PowerPreference, RequestAdapterOptions};

use crate::{Adapter, Instance, Surface};

#[cfg(any(not(target_arch = "wasm32"), feature = "wgc"))]
pub use wgc::instance::parse_backends_from_comma_list;
/// Always returns WEBGPU on wasm over webgpu.
#[cfg(all(target_arch = "wasm32", not(feature = "wgc")))]
pub fn parse_backends_from_comma_list(_string: &str) -> Backends {
    Backends::BROWSER_WEBGPU
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
            _ => return None,
        },
    )
}

/// Initialize the adapter obeying the WGPU_ADAPTER_NAME environment variable.
#[cfg(not(target_arch = "wasm32"))]
pub fn initialize_adapter_from_env(instance: &Instance, backend_bits: Backends) -> Option<Adapter> {
    let desired_adapter_name = std::env::var("WGPU_ADAPTER_NAME")
        .as_deref()
        .map(str::to_lowercase)
        .ok()?;

    let adapters = instance.enumerate_adapters(backend_bits);

    let mut chosen_adapter = None;
    for adapter in adapters {
        let info = adapter.get_info();

        if info.name.to_lowercase().contains(&desired_adapter_name) {
            chosen_adapter = Some(adapter);
            break;
        }
    }

    Some(chosen_adapter.expect("WGPU_ADAPTER_NAME set but no matching adapter found!"))
}

/// Initialize the adapter obeying the WGPU_ADAPTER_NAME environment variable.
#[cfg(target_arch = "wasm32")]
pub fn initialize_adapter_from_env(
    _instance: &Instance,
    _backend_bits: Backends,
) -> Option<Adapter> {
    None
}

/// Initialize the adapter obeying the WGPU_ADAPTER_NAME environment variable and if it doesn't exist fall back on a default adapter.
pub async fn initialize_adapter_from_env_or_default(
    instance: &Instance,
    backend_bits: wgt::Backends,
    compatible_surface: Option<&Surface>,
) -> Option<Adapter> {
    match initialize_adapter_from_env(instance, backend_bits) {
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
