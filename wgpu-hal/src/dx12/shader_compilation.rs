use std::ptr;

pub(super) use dxc::{compile_dxc, get_dxc_container, DxcContainer};
use winapi::um::d3dcompiler;

use crate::auxil::dxgi::result::HResult;

// This exists so that users who don't want to use dxc can disable the dxc_shader_compiler feature
// and not have to compile hassle_rs.
// Currently this will use Dxc if it is chosen as the dx12 compiler at `Instance` creation time, and will
// fallback to FXC if the Dxc libraries (dxil.dll and dxcompiler.dll) are not found, or if Fxc is chosen at'
// `Instance` creation time.

pub(super) fn compile_fxc(
    device: &super::Device,
    source: &String,
    source_name: &str,
    raw_ep: &std::ffi::CString,
    stage_bit: wgt::ShaderStages,
    full_stage: String,
) -> (
    Result<super::CompiledShader, crate::PipelineError>,
    log::Level,
) {
    profiling::scope!("compile_fxc");
    let mut shader_data = native::Blob::null();
    let mut compile_flags = d3dcompiler::D3DCOMPILE_ENABLE_STRICTNESS;
    if device
        .private_caps
        .instance_flags
        .contains(crate::InstanceFlags::DEBUG)
    {
        compile_flags |= d3dcompiler::D3DCOMPILE_DEBUG | d3dcompiler::D3DCOMPILE_SKIP_OPTIMIZATION;
    }
    let mut error = native::Blob::null();
    let hr = unsafe {
        profiling::scope!("d3dcompiler::D3DCompile");
        d3dcompiler::D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            source_name.as_ptr().cast(),
            ptr::null(),
            ptr::null_mut(),
            raw_ep.as_ptr(),
            full_stage.as_ptr().cast(),
            compile_flags,
            0,
            shader_data.mut_void().cast(),
            error.mut_void().cast(),
        )
    };

    match hr.into_result() {
        Ok(()) => (
            Ok(super::CompiledShader::Fxc(shader_data)),
            log::Level::Info,
        ),
        Err(e) => {
            let mut full_msg = format!("FXC D3DCompile error ({})", e);
            if !error.is_null() {
                use std::fmt::Write as _;
                let message = unsafe {
                    std::slice::from_raw_parts(
                        error.GetBufferPointer() as *const u8,
                        error.GetBufferSize(),
                    )
                };
                let _ = write!(full_msg, ": {}", String::from_utf8_lossy(message));
                unsafe {
                    error.destroy();
                }
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
    use std::path::PathBuf;

    pub(crate) struct DxcContainer {
        pub compiler: hassle_rs::DxcCompiler,
        pub library: hassle_rs::DxcLibrary,
        // Has to be held onto for the lifetime of the device otherwise shaders will fail to compile
        _dxc: hassle_rs::Dxc,
        // Only used when dxil.dll has an explicit path to avoid automatic validation silently failing.
        pub validator: Option<hassle_rs::DxcValidator>,
        _dxil: Option<hassle_rs::Dxil>,
    }

    pub(crate) fn get_dxc_container(
        dxc_path: Option<PathBuf>,
        dxil_path: Option<PathBuf>,
    ) -> Result<Option<DxcContainer>, crate::DeviceError> {
        // Make sure that dxil.dll exists.
        let dxil = match hassle_rs::Dxil::new(dxil_path.clone()) {
            Ok(dxil) => {
                if dxil_path.is_some() {
                    Some(dxil)
                } else {
                    None
                }
            }
            Err(e) => {
                log::warn!("Failed to load dxil.dll. Defaulting to Fxc instead: {}", e);
                return Ok(None);
            }
        };

        // We need to disable implicit validation and do explicit validation if the user is using
        // a custom path for `dxil.dll`, otherwise it will silently fail to do validation when
        // calling compile because dxcompiler is looking for `dxil.dll` in the local scope.
        let validator = if let Some(dxil) = &dxil {
            match dxil.create_validator() {
                Ok(validator) => Some(validator),
                Err(e) => {
                    log::warn!(
                        "Failed to create DXIL validator. Defaulting to Fxc instead: {}",
                        e
                    );
                    return Ok(None);
                }
            }
        } else {
            None
        };

        let dxc = match hassle_rs::Dxc::new(dxc_path) {
            Ok(dxc) => dxc,
            Err(e) => {
                log::warn!(
                    "Failed to load dxcompiler.dll. Defaulting to Fxc instead: {}",
                    e
                );
                return Ok(None);
            }
        };
        let dxc_compiler = dxc.create_compiler()?;
        let dxc_library = dxc.create_library()?;

        Ok(Some(DxcContainer {
            _dxc: dxc,
            compiler: dxc_compiler,
            library: dxc_library,
            _dxil: dxil,
            validator,
        }))
    }

    pub(crate) fn compile_dxc(
        device: &crate::dx12::Device,
        source: &str,
        source_name: &str,
        raw_ep: &str,
        stage_bit: wgt::ShaderStages,
        full_stage: String,
        dxc_container: &DxcContainer,
    ) -> (
        Result<crate::dx12::CompiledShader, crate::PipelineError>,
        log::Level,
    ) {
        profiling::scope!("compile_dxc");
        let mut compile_flags = arrayvec::ArrayVec::<&str, 4>::new_const();
        compile_flags.push("-Ges"); // d3dcompiler::D3DCOMPILE_ENABLE_STRICTNESS
        if device
            .private_caps
            .instance_flags
            .contains(crate::InstanceFlags::DEBUG)
        {
            compile_flags.push("-Zi"); // d3dcompiler::D3DCOMPILE_SKIP_OPTIMIZATION
            compile_flags.push("-Od"); // d3dcompiler::D3DCOMPILE_DEBUG
        }

        // Disable implicit validation if `dxil.dll` isn't available in the local scope, and
        // do explicit validation instead.
        if dxc_container.validator.is_some() {
            compile_flags.push("-Vd");
        }

        let blob = match dxc_container
            .library
            .create_blob_with_encoding_from_str(source)
            .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("DXC blob error: {}", e)))
        {
            Ok(blob) => blob,
            Err(e) => return (Err(e), log::Level::Error),
        };

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
                    // We have a blob, now check if we need to manually validate it or if it
                    // was automatically validated.
                    if let Some(validator) = &dxc_container.validator {
                        match validator.validate(dxc_blob) {
                            Ok(validated_blob) => (
                                Ok(crate::dx12::CompiledShader::Dxc(validated_blob.to_vec())),
                                log::Level::Info,
                            ),
                            Err(e) => (
                                Err(crate::PipelineError::Linkage(
                                    stage_bit,
                                    format!("DXC validation error: {:?}\n{:?}", e.0, e.1),
                                )),
                                log::Level::Error,
                            ),
                        }
                    } else {
                        // Automatically validated by dxc_container.compiler.compile()
                        (
                            Ok(crate::dx12::CompiledShader::Dxc(dxc_blob.to_vec())),
                            log::Level::Info,
                        )
                    }
                }
                Err(e) => (
                    Err(crate::PipelineError::Linkage(
                        stage_bit,
                        format!("DXC compile error: {}", e),
                    )),
                    log::Level::Error,
                ),
            },
            Err(e) => (
                Err(crate::PipelineError::Linkage(
                    stage_bit,
                    format!("DXC compile error: {:?}", e),
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
}

// These are stubs for when the `dxc_shader_compiler` feature is disabled.
#[cfg(not(feature = "dxc_shader_compiler"))]
mod dxc {
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
        _source_name: &str,
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
