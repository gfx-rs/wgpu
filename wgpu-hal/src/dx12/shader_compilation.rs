use std::ffi::CStr;
use std::ptr;

pub(super) use dxc::{compile_dxc, get_dxc_container, DxcContainer};
use windows::Win32::Graphics::Direct3D;

use crate::auxil::dxgi::result::HResult;

// This exists so that users who don't want to use dxc can disable the dxc_shader_compiler feature
// and not have to compile hassle_rs.
// Currently this will use Dxc if it is chosen as the dx12 compiler at `Instance` creation time, and will
// fallback to FXC if the Dxc libraries (dxil.dll and dxcompiler.dll) are not found, or if Fxc is chosen at'
// `Instance` creation time.

pub(super) fn compile_fxc(
    device: &super::Device,
    source: &str,
    source_name: Option<&CStr>,
    raw_ep: &CStr,
    stage_bit: wgt::ShaderStages,
    full_stage: &CStr,
) -> (
    Result<super::CompiledShader, crate::PipelineError>,
    log::Level,
) {
    profiling::scope!("compile_fxc");
    let mut shader_data = None;
    let mut compile_flags = Direct3D::Fxc::D3DCOMPILE_ENABLE_STRICTNESS;
    if device
        .private_caps
        .instance_flags
        .contains(wgt::InstanceFlags::DEBUG)
    {
        compile_flags |=
            Direct3D::Fxc::D3DCOMPILE_DEBUG | Direct3D::Fxc::D3DCOMPILE_SKIP_OPTIMIZATION;
    }

    // If no name has been set, D3DCompile wants the null pointer.
    let source_name = source_name.map(|cstr| cstr.as_ptr()).unwrap_or(ptr::null());

    let mut error = None;
    let hr = unsafe {
        profiling::scope!("Direct3D::Fxc::D3DCompile");
        Direct3D::Fxc::D3DCompile(
            // TODO: Update low-level bindings to accept a slice here
            source.as_ptr().cast(),
            source.len(),
            windows::core::PCSTR(source_name.cast()),
            None,
            None,
            windows::core::PCSTR(raw_ep.as_ptr().cast()),
            windows::core::PCSTR(full_stage.as_ptr().cast()),
            compile_flags,
            0,
            &mut shader_data,
            Some(&mut error),
        )
    };

    match hr.into_result() {
        Ok(()) => {
            let shader_data = shader_data.unwrap();
            (
                Ok(super::CompiledShader::Fxc(shader_data)),
                log::Level::Info,
            )
        }
        Err(e) => {
            let mut full_msg = format!("FXC D3DCompile error ({e})");
            if let Some(error) = error {
                use std::fmt::Write as _;
                let message = unsafe {
                    std::slice::from_raw_parts(
                        error.GetBufferPointer().cast(),
                        error.GetBufferSize(),
                    )
                };
                let _ = write!(full_msg, ": {}", String::from_utf8_lossy(message));
            }
            (
                Err(crate::PipelineError::Linkage(stage_bit, full_msg)),
                log::Level::Warn,
            )
        }
    }
}

// The Dxc implementation is behind a feature flag so that users who don't want to use dxc can disable the feature.
#[cfg(feature = "dxc_shader_compiler")]
mod dxc {
    use std::ffi::CStr;
    use std::path::PathBuf;

    // Destructor order should be fine since _dxil and _dxc don't rely on each other.
    pub(crate) struct DxcContainer {
        compiler: hassle_rs::DxcCompiler,
        library: hassle_rs::DxcLibrary,
        validator: hassle_rs::DxcValidator,
        // Has to be held onto for the lifetime of the device otherwise shaders will fail to compile.
        _dxc: hassle_rs::Dxc,
        // Also Has to be held onto for the lifetime of the device otherwise shaders will fail to validate.
        _dxil: hassle_rs::Dxil,
    }

    pub(crate) fn get_dxc_container(
        dxc_path: Option<PathBuf>,
        dxil_path: Option<PathBuf>,
    ) -> Result<Option<DxcContainer>, crate::DeviceError> {
        // Make sure that dxil.dll exists.
        let dxil = match hassle_rs::Dxil::new(dxil_path) {
            Ok(dxil) => dxil,
            Err(e) => {
                log::warn!("Failed to load dxil.dll. Defaulting to FXC instead: {}", e);
                return Ok(None);
            }
        };

        // Needed for explicit validation.
        let validator = dxil.create_validator()?;

        let dxc = match hassle_rs::Dxc::new(dxc_path) {
            Ok(dxc) => dxc,
            Err(e) => {
                log::warn!(
                    "Failed to load dxcompiler.dll. Defaulting to FXC instead: {}",
                    e
                );
                return Ok(None);
            }
        };
        let compiler = dxc.create_compiler()?;
        let library = dxc.create_library()?;

        Ok(Some(DxcContainer {
            _dxc: dxc,
            compiler,
            library,
            _dxil: dxil,
            validator,
        }))
    }

    pub(crate) fn compile_dxc(
        device: &crate::dx12::Device,
        source: &str,
        source_name: Option<&CStr>,
        raw_ep: &str,
        stage_bit: wgt::ShaderStages,
        full_stage: String,
        dxc_container: &DxcContainer,
    ) -> (
        Result<crate::dx12::CompiledShader, crate::PipelineError>,
        log::Level,
    ) {
        profiling::scope!("compile_dxc");
        let mut compile_flags = arrayvec::ArrayVec::<&str, 6>::new_const();
        compile_flags.push("-Ges"); // Direct3D::Fxc::D3DCOMPILE_ENABLE_STRICTNESS
        compile_flags.push("-Vd"); // Disable implicit validation to work around bugs when dxil.dll isn't in the local directory.
        compile_flags.push("-HV"); // Use HLSL 2018, Naga doesn't supported 2021 yet.
        compile_flags.push("2018");

        if device
            .private_caps
            .instance_flags
            .contains(wgt::InstanceFlags::DEBUG)
        {
            compile_flags.push("-Zi"); // Direct3D::Fxc::D3DCOMPILE_SKIP_OPTIMIZATION
            compile_flags.push("-Od"); // Direct3D::Fxc::D3DCOMPILE_DEBUG
        }

        let blob = match dxc_container
            .library
            .create_blob_with_encoding_from_str(source)
            .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("DXC blob error: {e}")))
        {
            Ok(blob) => blob,
            Err(e) => return (Err(e), log::Level::Error),
        };

        let source_name = source_name
            .and_then(|cstr| cstr.to_str().ok())
            .unwrap_or("");

        let compiled = dxc_container.compiler.compile(
            &blob,
            source_name,
            raw_ep,
            &full_stage,
            &compile_flags,
            None,
            &[],
        );

        let (result, log_level) = match compiled {
            Ok(dxc_result) => match dxc_result.get_result() {
                Ok(dxc_blob) => {
                    // Validate the shader.
                    match dxc_container.validator.validate(dxc_blob) {
                        Ok(validated_blob) => (
                            Ok(crate::dx12::CompiledShader::Dxc(validated_blob.to_vec())),
                            log::Level::Info,
                        ),
                        Err(e) => (
                            Err(crate::PipelineError::Linkage(
                                stage_bit,
                                format!(
                                    "DXC validation error: {:?}\n{:?}",
                                    get_error_string_from_dxc_result(&dxc_container.library, &e.0)
                                        .unwrap_or_default(),
                                    e.1
                                ),
                            )),
                            log::Level::Error,
                        ),
                    }
                }
                Err(e) => (
                    Err(crate::PipelineError::Linkage(
                        stage_bit,
                        format!("DXC compile error: {e}"),
                    )),
                    log::Level::Error,
                ),
            },
            Err(e) => (
                Err(crate::PipelineError::Linkage(
                    stage_bit,
                    format!(
                        "DXC compile error: {}",
                        get_error_string_from_dxc_result(&dxc_container.library, &e.0)
                            .unwrap_or_default()
                    ),
                )),
                log::Level::Error,
            ),
        };

        (result, log_level)
    }

    impl From<hassle_rs::HassleError> for crate::DeviceError {
        fn from(value: hassle_rs::HassleError) -> Self {
            match value {
                hassle_rs::HassleError::Win32Error(e) => {
                    // TODO: This returns an HRESULT, should we try and use the associated Windows error message?
                    log::error!("Win32 error: {e:?}");
                    crate::DeviceError::Lost
                }
                hassle_rs::HassleError::LoadLibraryError { filename, inner } => {
                    log::error!("Failed to load dxc library {filename:?}. Inner error: {inner:?}");
                    crate::DeviceError::Lost
                }
                hassle_rs::HassleError::LibLoadingError(e) => {
                    log::error!("Failed to load dxc library. {e:?}");
                    crate::DeviceError::Lost
                }
                hassle_rs::HassleError::WindowsOnly(e) => {
                    log::error!("Signing with dxil.dll is only supported on Windows. {e:?}");
                    crate::DeviceError::Lost
                }
                // `ValidationError` and `CompileError` should never happen in a context involving `DeviceError`
                hassle_rs::HassleError::ValidationError(_e) => unimplemented!(),
                hassle_rs::HassleError::CompileError(_e) => unimplemented!(),
            }
        }
    }

    fn get_error_string_from_dxc_result(
        library: &hassle_rs::DxcLibrary,
        error: &hassle_rs::DxcOperationResult,
    ) -> Result<String, hassle_rs::HassleError> {
        error
            .get_error_buffer()
            .and_then(|error| library.get_blob_as_string(&hassle_rs::DxcBlob::from(error)))
    }
}

// These are stubs for when the `dxc_shader_compiler` feature is disabled.
#[cfg(not(feature = "dxc_shader_compiler"))]
mod dxc {
    use std::ffi::CStr;
    use std::path::PathBuf;

    pub(crate) struct DxcContainer {}

    pub(crate) fn get_dxc_container(
        _dxc_path: Option<PathBuf>,
        _dxil_path: Option<PathBuf>,
    ) -> Result<Option<DxcContainer>, crate::DeviceError> {
        // Falls back to Fxc and logs an error.
        log::error!("DXC shader compiler was requested on Instance creation, but the DXC feature is disabled. Enable the `dxc_shader_compiler` feature on wgpu_hal to use DXC.");
        Ok(None)
    }

    // It shouldn't be possible that this gets called with the `dxc_shader_compiler` feature disabled.
    pub(crate) fn compile_dxc(
        _device: &crate::dx12::Device,
        _source: &str,
        _source_name: Option<&CStr>,
        _raw_ep: &str,
        _stage_bit: wgt::ShaderStages,
        _full_stage: String,
        _dxc_container: &DxcContainer,
    ) -> (
        Result<crate::dx12::CompiledShader, crate::PipelineError>,
        log::Level,
    ) {
        unimplemented!("Something went really wrong, please report this. Attempted to compile shader with DXC, but the DXC feature is disabled. Enable the `dxc_shader_compiler` feature on wgpu_hal to use DXC.");
    }
}
