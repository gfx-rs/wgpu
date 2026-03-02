use alloc::sync::Arc;
use core::ops::Deref;

use windows::Win32::Graphics::{Direct3D, Direct3D12};

use super::D3D12Lib;
use crate::auxil::dxgi::factory::DxgiAdapter;

/// Abstraction over D3D12 device creation.
///
/// Supports two paths:
/// - **Independent**: Uses `ID3D12DeviceFactory` from the Agility SDK's Independent Devices API.
/// - **Legacy**: Uses the traditional `D3D12CreateDevice` export.
pub(super) enum DeviceFactory {
    /// Uses `ID3D12DeviceFactory` from the Independent Devices API.
    Independent(Direct3D12::ID3D12DeviceFactory),
    /// Uses the traditional `D3D12CreateDevice` export.
    Legacy,
}

impl DeviceFactory {
    /// Create a new `DeviceFactory`.
    ///
    /// If `agility_sdk` is `Some`, attempts to set up the Independent Devices API path.
    /// On failure, the behavior depends on
    /// [`on_load_failure`](wgt::Dx12AgilitySDKLoadFailure):
    /// - [`Fallback`](wgt::Dx12AgilitySDKLoadFailure::Fallback): logs a warning and
    ///   returns `Ok(Legacy)`.
    /// - [`Error`](wgt::Dx12AgilitySDKLoadFailure::Error): returns an `Err`.
    pub(super) fn new(
        lib: &D3D12Lib,
        agility_sdk: Option<&wgt::Dx12AgilitySDK>,
    ) -> Result<Self, crate::InstanceError> {
        let Some(agility_sdk) = agility_sdk else {
            log::info!("No D3D12 Agility SDK configuration provided; using system D3D12 runtime");
            return Ok(Self::Legacy);
        };

        match Self::try_create_independent(lib, agility_sdk) {
            Ok(factory) => {
                log::info!(
                    "Using D3D12 Agility SDK v{} from '{}'",
                    agility_sdk.sdk_version,
                    agility_sdk.sdk_path
                );
                Ok(Self::Independent(factory))
            }
            Err(err) => {
                let message = format!(
                    "Failed to initialize D3D12 Agility SDK (v{} at '{}'): {err}",
                    agility_sdk.sdk_version, agility_sdk.sdk_path
                );

                match agility_sdk.on_load_failure {
                    wgt::Dx12AgilitySDKLoadFailure::Fallback => {
                        log::warn!("{message}; falling back to system D3D12 runtime");
                        Ok(Self::Legacy)
                    }
                    wgt::Dx12AgilitySDKLoadFailure::Error => {
                        Err(crate::InstanceError::new(message))
                    }
                }
            }
        }
    }

    fn try_create_independent(
        lib: &D3D12Lib,
        agility_sdk: &wgt::Dx12AgilitySDK,
    ) -> Result<Direct3D12::ID3D12DeviceFactory, DeviceFactoryError> {
        // Step 1: Get ID3D12SDKConfiguration1 via D3D12GetInterface
        let sdk_config: Direct3D12::ID3D12SDKConfiguration1 = lib
            .get_interface(&Direct3D12::CLSID_D3D12SDKConfiguration)
            .map_err(DeviceFactoryError::GetInterface)?;

        // Step 2: Create device factory with the specified SDK version and path
        let sdk_path = std::ffi::CString::new(agility_sdk.sdk_path.as_bytes())
            .map_err(|_| DeviceFactoryError::InvalidPath)?;
        let factory: Direct3D12::ID3D12DeviceFactory = unsafe {
            sdk_config.CreateDeviceFactory(
                agility_sdk.sdk_version,
                windows::core::PCSTR(sdk_path.as_ptr().cast::<u8>()),
            )
        }
        .map_err(DeviceFactoryError::CreateDeviceFactory)?;

        Ok(factory)
    }

    /// Initialize the device factory from global state.
    ///
    /// This should be called after the debug layer is enabled so the factory
    /// picks up debug settings. Only meaningful for the Independent path.
    pub(super) fn initialize_from_global_state(&self) {
        if let Self::Independent(factory) = self {
            if let Err(err) = unsafe { factory.InitializeFromGlobalState() } {
                log::warn!("ID3D12DeviceFactory::InitializeFromGlobalState failed: {err}");
            }
        }
    }

    /// Create a D3D12 device using the appropriate method.
    pub(super) fn create_device(
        &self,
        lib: &Arc<D3D12Lib>,
        adapter: &DxgiAdapter,
        feature_level: Direct3D::D3D_FEATURE_LEVEL,
    ) -> Result<Direct3D12::ID3D12Device, super::CreateDeviceError> {
        match self {
            Self::Independent(factory) => {
                let mut result__: Option<Direct3D12::ID3D12Device> = None;
                unsafe { factory.CreateDevice(adapter.deref(), feature_level, &mut result__) }
                    .map_err(|e| super::CreateDeviceError::D3D12CreateDevice(e.into()))?;

                result__.ok_or(super::CreateDeviceError::RetDeviceIsNull)
            }
            Self::Legacy => lib.create_device(adapter, feature_level),
        }
    }
}

impl core::fmt::Debug for DeviceFactory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Independent(_) => write!(f, "DeviceFactory::Independent"),
            Self::Legacy => write!(f, "DeviceFactory::Legacy"),
        }
    }
}

#[derive(Debug, thiserror::Error)]
enum DeviceFactoryError {
    #[error("failed to get ID3D12SDKConfiguration1: {0}")]
    GetInterface(super::GetInterfaceError),
    #[error("SDK path contains null bytes")]
    InvalidPath,
    #[error("CreateDeviceFactory failed: {0}")]
    CreateDeviceFactory(windows::core::Error),
}
