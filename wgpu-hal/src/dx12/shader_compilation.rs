use crate::auxil::dxgi::result::HResult;
use std::ffi::CStr;
use std::path::PathBuf;
use windows::{
    core::{Interface, PCSTR, PCWSTR},
    Win32::Graphics::Direct3D::{Dxc, Fxc},
};

// Currently this will use Dxc if it is chosen as the dx12 compiler at `Instance` creation time, and will
// fallback to FXC if the Dxc libraries (dxil.dll and dxcompiler.dll) are not found, or if Fxc is chosen at'
// `Instance` creation time.

pub(super) fn compile_fxc(
    device: &super::Device,
    source: &str,
    source_name: Option<&CStr>,
    raw_ep: &str,
    stage_bit: wgt::ShaderStages,
    full_stage: &str,
) -> Result<super::CompiledShader, crate::PipelineError> {
    profiling::scope!("compile_fxc");
    let mut shader_data = None;
    let mut compile_flags = Fxc::D3DCOMPILE_ENABLE_STRICTNESS;
    if device
        .private_caps
        .instance_flags
        .contains(wgt::InstanceFlags::DEBUG)
    {
        compile_flags |= Fxc::D3DCOMPILE_DEBUG | Fxc::D3DCOMPILE_SKIP_OPTIMIZATION;
    }

    let raw_ep = std::ffi::CString::new(raw_ep).unwrap();
    let full_stage = std::ffi::CString::new(full_stage).unwrap();

    // If no name has been set, D3DCompile wants the null pointer.
    let source_name = source_name
        .map(|cstr| cstr.as_ptr().cast())
        .unwrap_or(core::ptr::null());

    let mut error = None;
    let hr = unsafe {
        profiling::scope!("Fxc::D3DCompile");
        Fxc::D3DCompile(
            // TODO: Update low-level bindings to accept a slice here
            source.as_ptr().cast(),
            source.len(),
            PCSTR(source_name),
            None,
            None,
            PCSTR(raw_ep.as_ptr().cast()),
            PCSTR(full_stage.as_ptr().cast()),
            compile_flags,
            0,
            &mut shader_data,
            Some(&mut error),
        )
    };

    match hr {
        Ok(()) => {
            let shader_data = shader_data.unwrap();
            Ok(super::CompiledShader::Fxc(shader_data))
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
            Err(crate::PipelineError::Linkage(stage_bit, full_msg))
        }
    }
}

trait DxcObj: Interface {
    const CLSID: windows::core::GUID;
}
impl DxcObj for Dxc::IDxcCompiler3 {
    const CLSID: windows::core::GUID = Dxc::CLSID_DxcCompiler;
}
impl DxcObj for Dxc::IDxcUtils {
    const CLSID: windows::core::GUID = Dxc::CLSID_DxcUtils;
}
impl DxcObj for Dxc::IDxcValidator {
    const CLSID: windows::core::GUID = Dxc::CLSID_DxcValidator;
}

#[derive(Debug)]
struct DxcLib {
    lib: crate::dx12::DynLib,
}

impl DxcLib {
    fn new(lib_path: Option<PathBuf>, lib_name: &'static str) -> Result<Self, libloading::Error> {
        let lib_path = if let Some(lib_path) = lib_path {
            if lib_path.is_file() {
                lib_path
            } else {
                lib_path.join(lib_name)
            }
        } else {
            PathBuf::from(lib_name)
        };
        unsafe { crate::dx12::DynLib::new(lib_path).map(|lib| Self { lib }) }
    }

    pub fn create_instance<T: DxcObj>(&self) -> Result<T, crate::DeviceError> {
        type Fun = extern "system" fn(
            rclsid: *const windows_core::GUID,
            riid: *const windows_core::GUID,
            ppv: *mut *mut core::ffi::c_void,
        ) -> windows_core::HRESULT;
        let func: libloading::Symbol<Fun> = unsafe { self.lib.get(b"DxcCreateInstance\0") }?;

        let mut result__ = None;
        (func)(&T::CLSID, &T::IID, <*mut _>::cast(&mut result__))
            .ok()
            .into_device_result("DxcCreateInstance")?;
        result__.ok_or(crate::DeviceError::Unexpected)
    }
}

// Destructor order should be fine since _dxil and _dxc don't rely on each other.
pub(super) struct DxcContainer {
    compiler: Dxc::IDxcCompiler3,
    utils: Dxc::IDxcUtils,
    validator: Dxc::IDxcValidator,
    // Has to be held onto for the lifetime of the device otherwise shaders will fail to compile.
    _dxc: DxcLib,
    // Also Has to be held onto for the lifetime of the device otherwise shaders will fail to validate.
    _dxil: DxcLib,
}

pub(super) fn get_dxc_container(
    dxc_path: Option<PathBuf>,
    dxil_path: Option<PathBuf>,
) -> Result<Option<DxcContainer>, crate::DeviceError> {
    let dxc = match DxcLib::new(dxc_path, "dxcompiler.dll") {
        Ok(dxc) => dxc,
        Err(e) => {
            log::warn!(
                "Failed to load dxcompiler.dll. Defaulting to FXC instead: {}",
                e
            );
            return Ok(None);
        }
    };

    let dxil = match DxcLib::new(dxil_path, "dxil.dll") {
        Ok(dxil) => dxil,
        Err(e) => {
            log::warn!("Failed to load dxil.dll. Defaulting to FXC instead: {}", e);
            return Ok(None);
        }
    };

    let compiler = dxc.create_instance::<Dxc::IDxcCompiler3>()?;
    let utils = dxc.create_instance::<Dxc::IDxcUtils>()?;
    let validator = dxil.create_instance::<Dxc::IDxcValidator>()?;

    Ok(Some(DxcContainer {
        compiler,
        utils,
        validator,
        _dxc: dxc,
        _dxil: dxil,
    }))
}

/// Owned PCWSTR
#[allow(clippy::upper_case_acronyms)]
struct OPCWSTR {
    inner: Vec<u16>,
}

impl OPCWSTR {
    fn new(s: &str) -> Self {
        let mut inner: Vec<_> = s.encode_utf16().collect();
        inner.push(0);
        Self { inner }
    }

    fn ptr(&self) -> PCWSTR {
        PCWSTR(self.inner.as_ptr())
    }
}

fn get_output<T: Interface>(
    res: &Dxc::IDxcResult,
    kind: Dxc::DXC_OUT_KIND,
) -> Result<T, crate::DeviceError> {
    let mut result__: Option<T> = None;
    unsafe { res.GetOutput::<T>(kind, &mut None, <*mut _>::cast(&mut result__)) }
        .into_device_result("GetOutput")?;
    result__.ok_or(crate::DeviceError::Unexpected)
}

fn as_err_str(blob: &Dxc::IDxcBlobUtf8) -> Result<&str, crate::DeviceError> {
    let ptr = unsafe { blob.GetStringPointer() };
    let len = unsafe { blob.GetStringLength() };
    core::str::from_utf8(unsafe { core::slice::from_raw_parts(ptr.0, len) })
        .map_err(|_| crate::DeviceError::Unexpected)
}

pub(super) fn compile_dxc(
    device: &crate::dx12::Device,
    source: &str,
    source_name: Option<&CStr>,
    raw_ep: &str,
    stage_bit: wgt::ShaderStages,
    full_stage: &str,
    dxc_container: &DxcContainer,
) -> Result<crate::dx12::CompiledShader, crate::PipelineError> {
    profiling::scope!("compile_dxc");

    let source_name = source_name.and_then(|cstr| cstr.to_str().ok());

    let source_name = source_name.map(OPCWSTR::new);
    let raw_ep = OPCWSTR::new(raw_ep);
    let full_stage = OPCWSTR::new(full_stage);

    let mut compile_args = arrayvec::ArrayVec::<PCWSTR, 12>::new_const();

    if let Some(source_name) = source_name.as_ref() {
        compile_args.push(source_name.ptr())
    }

    compile_args.extend([
        windows::core::w!("-E"),
        raw_ep.ptr(),
        windows::core::w!("-T"),
        full_stage.ptr(),
        windows::core::w!("-HV"),
        windows::core::w!("2018"), // Use HLSL 2018, Naga doesn't supported 2021 yet.
        windows::core::w!("-no-warnings"),
        Dxc::DXC_ARG_ENABLE_STRICTNESS,
        Dxc::DXC_ARG_SKIP_VALIDATION, // Disable implicit validation to work around bugs when dxil.dll isn't in the local directory.
    ]);

    if device
        .private_caps
        .instance_flags
        .contains(wgt::InstanceFlags::DEBUG)
    {
        compile_args.push(Dxc::DXC_ARG_DEBUG);
        compile_args.push(Dxc::DXC_ARG_SKIP_OPTIMIZATIONS);
    }

    let buffer = Dxc::DxcBuffer {
        Ptr: source.as_ptr().cast(),
        Size: source.len(),
        Encoding: Dxc::DXC_CP_UTF8.0,
    };

    let compile_res: Dxc::IDxcResult = unsafe {
        dxc_container
            .compiler
            .Compile(&buffer, Some(&compile_args), None)
    }
    .into_device_result("Compile")?;

    drop(compile_args);
    drop(source_name);
    drop(raw_ep);
    drop(full_stage);

    let err_blob = get_output::<Dxc::IDxcBlobUtf8>(&compile_res, Dxc::DXC_OUT_ERRORS)?;

    let len = unsafe { err_blob.GetStringLength() };
    if len != 0 {
        let err = as_err_str(&err_blob)?;
        return Err(crate::PipelineError::Linkage(
            stage_bit,
            format!("DXC compile error: {err}"),
        ));
    }

    let blob = get_output::<Dxc::IDxcBlob>(&compile_res, Dxc::DXC_OUT_OBJECT)?;

    let err_blob = {
        let res = unsafe {
            dxc_container
                .validator
                .Validate(&blob, Dxc::DxcValidatorFlags_InPlaceEdit)
        }
        .into_device_result("Validate")?;

        unsafe { res.GetErrorBuffer() }.into_device_result("GetErrorBuffer")?
    };

    let size = unsafe { err_blob.GetBufferSize() };
    if size != 0 {
        let err_blob = unsafe { dxc_container.utils.GetBlobAsUtf8(&err_blob) }
            .into_device_result("GetBlobAsUtf8")?;
        let err = as_err_str(&err_blob)?;
        return Err(crate::PipelineError::Linkage(
            stage_bit,
            format!("DXC validation error: {err}"),
        ));
    }

    Ok(crate::dx12::CompiledShader::Dxc(blob))
}
