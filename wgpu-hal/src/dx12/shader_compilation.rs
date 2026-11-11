use alloc::{string::String, vec::Vec};
use core::ffi::CStr;
use std::path::PathBuf;

use crate::auxil::dxgi::result::HResult;
use thiserror::Error;
use windows::{
    core::{Interface, PCSTR, PCWSTR},
    Win32::Graphics::Direct3D::{Dxc, Fxc, ID3DBlob, D3D_SHADER_MACRO},
};

pub(super) enum CompilerContainer {
    Fxc(CompilerFxc),
    DynamicDxc(CompilerDynamicDxc),
    #[cfg_attr(not(static_dxc), allow(unused))]
    StaticDxc(CompilerStaticDxc),
}

pub(super) struct CompilerFxc {
    fxc: FxcLib,
}

pub(super) struct CompilerDynamicDxc {
    denormal_mode: Option<wgt::DxcDenormalMode>,
    max_shader_model: wgt::DxcShaderModel,
    compiler: Dxc::IDxcCompiler3,
    // Has to be held onto for the lifetime of the device otherwise shaders will fail to compile.
    // Only needed when using dynamic linking.
    _dxc: DxcLib,
}

pub(super) struct CompilerStaticDxc {
    denormal_mode: Option<wgt::DxcDenormalMode>,
    max_shader_model: wgt::DxcShaderModel,
    compiler: Dxc::IDxcCompiler3,
}

#[derive(Debug, Error)]
pub(super) enum GetContainerError {
    #[error(transparent)]
    Device(#[from] crate::DeviceError),
    #[error("Failed to load {0}: {1}")]
    FailedToLoad(&'static str, libloading::Error),
}

impl CompilerContainer {
    pub(super) fn with_denormal_mode(
        mut self,
        mode: Option<wgt::DxcDenormalMode>,
    ) -> Result<Self, crate::InstanceError> {
        match &mut self {
            Self::Fxc(_) if mode.is_some() => {
                return Err(crate::InstanceError::new(String::from(
                    "An explicit DXC denormal mode requires DXC; FXC was selected",
                )));
            }
            Self::Fxc(_) => {}
            Self::DynamicDxc(compiler) => compiler.denormal_mode = mode,
            Self::StaticDxc(compiler) => compiler.denormal_mode = mode,
        }
        Ok(self)
    }

    pub(super) fn denormal_mode(&self) -> Option<wgt::DxcDenormalMode> {
        match self {
            Self::Fxc(_) => None,
            Self::DynamicDxc(compiler) => compiler.denormal_mode,
            Self::StaticDxc(compiler) => compiler.denormal_mode,
        }
    }

    pub(super) fn new_fxc() -> Result<Self, GetContainerError> {
        FxcLib::new_dynamic().map(|fxc| Self::Fxc(CompilerFxc { fxc }))
    }

    pub(super) fn new_dynamic_dxc(dxc_path: PathBuf) -> Result<Self, GetContainerError> {
        let dxc = DxcLib::new_dynamic(dxc_path)
            .map_err(|e| GetContainerError::FailedToLoad("dxcompiler.dll", e))?;

        let compiler = dxc.create_instance::<Dxc::IDxcCompiler3>()?;

        let (mut major, mut minor) = (1, 0);
        // DXC 1.0 didn't support this. If the cast fails assume it is DXC 1.0.
        if let Ok(version_info) = compiler.cast::<Dxc::IDxcVersionInfo>() {
            unsafe {
                version_info.GetVersion(&mut major, &mut minor).unwrap();
            }
        }

        Ok(Self::DynamicDxc(CompilerDynamicDxc {
            max_shader_model: wgt::DxcShaderModel::from_dxc_version(major, minor),
            denormal_mode: None,
            compiler,
            _dxc: dxc,
        }))
    }

    /// Creates a [`CompilerContainer`] that delegates to the statically-linked version of DXC.
    pub(super) fn new_static_dxc() -> Result<CompilerContainer, crate::DeviceError> {
        #[cfg(static_dxc)]
        {
            unsafe {
                let compiler = dxc_create_instance::<Dxc::IDxcCompiler3>(|clsid, iid, ppv| {
                    windows_core::HRESULT(mach_dxcompiler_rs::DxcCreateInstance(
                        clsid.cast(),
                        iid.cast(),
                        ppv,
                    ))
                })?;

                Ok(CompilerContainer::StaticDxc(CompilerStaticDxc {
                    max_shader_model: wgt::DxcShaderModel::V6_7,
                    denormal_mode: None,
                    compiler,
                }))
            }
        }
        #[cfg(not(static_dxc))]
        {
            panic!("Attempted to create a static DXC shader compiler, but the static-dxc feature was not enabled")
        }
    }

    pub(super) fn max_shader_model(&self) -> Option<wgt::DxcShaderModel> {
        match *self {
            CompilerContainer::Fxc(..) => None,
            CompilerContainer::DynamicDxc(CompilerDynamicDxc {
                max_shader_model, ..
            })
            | CompilerContainer::StaticDxc(CompilerStaticDxc {
                max_shader_model, ..
            }) => Some(max_shader_model),
        }
    }

    pub(super) fn compile(
        &self,
        device: &super::Device,
        key: &super::ShaderCacheKey,
        source_name: Option<&CStr>,
        stage_bit: wgt::ShaderStages,
        full_stage: &str,
    ) -> Result<super::CompiledShader, crate::PipelineError> {
        validate_denormal_mode(self.denormal_mode(), key.shader_model)
            .map_err(|message| crate::PipelineError::Linkage(stage_bit, message.into()))?;
        match self {
            CompilerContainer::Fxc(CompilerFxc { fxc }) => compile_fxc(
                device,
                &key.source,
                source_name,
                &key.entry_point,
                stage_bit,
                full_stage,
                fxc,
            ),
            CompilerContainer::DynamicDxc(CompilerDynamicDxc { compiler, .. })
            | CompilerContainer::StaticDxc(CompilerStaticDxc { compiler, .. }) => compile_dxc(
                device,
                &key.source,
                source_name,
                &key.entry_point,
                stage_bit,
                full_stage,
                compiler,
                self.denormal_mode(),
            ),
        }
    }
}

type D3DCompileFn = unsafe extern "system" fn(
    psrcdata: *const core::ffi::c_void,
    srcdatasize: usize,
    psourcename: PCSTR,
    pdefines: *const D3D_SHADER_MACRO,
    pinclude: *mut core::ffi::c_void,
    pentrypoint: PCSTR,
    ptarget: PCSTR,
    flags1: u32,
    flags2: u32,
    ppcode: *mut *mut core::ffi::c_void,
    pperrormsgs: *mut *mut core::ffi::c_void,
) -> windows_core::HRESULT;

#[derive(Debug)]
struct FxcLib {
    // `d3dcompile_fn` points into `_lib`, so `_lib` must be held for as long
    // as we want to keep compiling shaders with FXC.
    _lib: crate::dx12::DynLib,
    d3dcompile_fn: D3DCompileFn,
}

impl FxcLib {
    const PATH: &str = "d3dcompiler_47.dll";

    fn new_dynamic() -> Result<Self, GetContainerError> {
        unsafe {
            let lib = crate::dx12::DynLib::new(Self::PATH)
                .map_err(|e| GetContainerError::FailedToLoad(FxcLib::PATH, e))?;
            let d3dcompile_fn: D3DCompileFn = *lib.get::<D3DCompileFn>(c"D3DCompile".to_bytes())?;

            Ok(Self {
                _lib: lib,
                d3dcompile_fn,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compile(
        &self,
        source: &str,
        source_name: Option<&CStr>,
        raw_ep: &str,
        full_stage: &str,
        compile_flags: u32,
        shader_data: &mut Option<ID3DBlob>,
        error: &mut Option<ID3DBlob>,
    ) -> Result<windows_core::Result<()>, crate::DeviceError> {
        unsafe {
            let raw_ep = alloc::ffi::CString::new(raw_ep).unwrap();
            let full_stage = alloc::ffi::CString::new(full_stage).unwrap();

            // If no name has been set, D3DCompile wants the null pointer.
            let source_name = source_name
                .map(|cstr| cstr.as_ptr().cast())
                .unwrap_or(core::ptr::null());

            let shader_data: *mut Option<ID3DBlob> = shader_data;
            let error: *mut Option<ID3DBlob> = error;

            {
                profiling::scope!("Fxc::D3DCompile");
                Ok((self.d3dcompile_fn)(
                    source.as_ptr().cast(),
                    source.len(),
                    PCSTR(source_name),
                    core::ptr::null(),
                    core::ptr::null_mut(),
                    PCSTR(raw_ep.as_ptr().cast()),
                    PCSTR(full_stage.as_ptr().cast()),
                    compile_flags,
                    0,
                    shader_data.cast(),
                    error.cast(),
                )
                .ok())
            }
        }
    }
}

fn compile_fxc(
    device: &super::Device,
    source: &str,
    source_name: Option<&CStr>,
    raw_ep: &str,
    stage_bit: wgt::ShaderStages,
    full_stage: &str,
    fxc: &FxcLib,
) -> Result<super::CompiledShader, crate::PipelineError> {
    profiling::scope!("compile_fxc");
    let mut compile_flags = Fxc::D3DCOMPILE_ENABLE_STRICTNESS;
    if device
        .shared
        .private_caps
        .instance_flags
        .contains(wgt::InstanceFlags::DEBUG)
    {
        compile_flags |= Fxc::D3DCOMPILE_DEBUG | Fxc::D3DCOMPILE_SKIP_OPTIMIZATION;
    }

    let mut shader_data = None;
    let mut error = None;
    let hr = fxc.compile(
        source,
        source_name,
        raw_ep,
        full_stage,
        compile_flags,
        &mut shader_data,
        &mut error,
    )?;

    match hr {
        Ok(()) => {
            let shader_data = super::D3DBlob(shader_data.unwrap());
            Ok(unsafe { shader_data.as_slice() }.into())
        }
        Err(e) => {
            let mut full_msg = format!("FXC D3DCompile error ({e})");
            if let Some(error) = error {
                use core::fmt::Write as _;
                let message = unsafe {
                    core::slice::from_raw_parts(
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
    fn new_dynamic(lib_path: PathBuf) -> Result<Self, libloading::Error> {
        unsafe { crate::dx12::DynLib::new(lib_path).map(|lib| Self { lib }) }
    }

    pub fn create_instance<T: DxcObj>(&self) -> Result<T, crate::DeviceError> {
        unsafe {
            type DxcCreateInstanceFn = unsafe extern "system" fn(
                rclsid: *const windows_core::GUID,
                riid: *const windows_core::GUID,
                ppv: *mut *mut core::ffi::c_void,
            )
                -> windows_core::HRESULT;

            let func: libloading::Symbol<DxcCreateInstanceFn> =
                self.lib.get(c"DxcCreateInstance".to_bytes())?;
            dxc_create_instance::<T>(|clsid, iid, ppv| func(clsid, iid, ppv))
        }
    }
}

/// Invokes the provided library function to create a DXC object.
unsafe fn dxc_create_instance<T: DxcObj>(
    f: impl Fn(
        *const windows_core::GUID,
        *const windows_core::GUID,
        *mut *mut core::ffi::c_void,
    ) -> windows_core::HRESULT,
) -> Result<T, crate::DeviceError> {
    let mut result__ = None;
    f(&T::CLSID, &T::IID, <*mut _>::cast(&mut result__))
        .ok()
        .into_device_result("DxcCreateInstance")?;
    result__.ok_or(crate::DeviceError::Unexpected)
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

// Validate the actual selected profile, not just the compiler's maximum profile.
fn validate_denormal_mode(
    mode: Option<wgt::DxcDenormalMode>,
    shader_model: naga::back::hlsl::ShaderModel,
) -> Result<(), &'static str> {
    if mode.is_some() && shader_model < naga::back::hlsl::ShaderModel::V6_2 {
        Err("An explicit DXC denormal mode requires shader model 6.2 or later")
    } else {
        Ok(())
    }
}

// The caller keeps the three owned UTF-16 strings alive through Compile. All other
// pointers refer to static strings. Leave the existing argument order intact for None.
fn dxc_arguments(
    source_name: Option<&OPCWSTR>,
    raw_ep: &OPCWSTR,
    full_stage: &OPCWSTR,
    debug: bool,
    enable_16bit: bool,
    denormal_mode: Option<wgt::DxcDenormalMode>,
) -> arrayvec::ArrayVec<PCWSTR, 15> {
    let mut compile_args = arrayvec::ArrayVec::<PCWSTR, 15>::new_const();
    if let Some(source_name) = source_name {
        compile_args.push(source_name.ptr());
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
    ]);
    if debug {
        compile_args.push(Dxc::DXC_ARG_DEBUG);
        compile_args.push(Dxc::DXC_ARG_SKIP_OPTIMIZATIONS);
    }
    if enable_16bit {
        compile_args.push(windows::core::w!("-enable-16bit-types"));
    }
    if let Some(mode) = denormal_mode {
        compile_args.extend([
            windows::core::w!("-denorm"),
            match mode {
                wgt::DxcDenormalMode::Any => windows::core::w!("any"),
                wgt::DxcDenormalMode::Preserve => windows::core::w!("preserve"),
                wgt::DxcDenormalMode::FlushToZero => windows::core::w!("ftz"),
            },
        ]);
    }
    compile_args
}

#[allow(clippy::too_many_arguments)]
fn compile_dxc(
    device: &crate::dx12::Device,
    source: &str,
    source_name: Option<&CStr>,
    raw_ep: &str,
    stage_bit: wgt::ShaderStages,
    full_stage: &str,
    compiler: &Dxc::IDxcCompiler3,
    denormal_mode: Option<wgt::DxcDenormalMode>,
) -> Result<crate::dx12::CompiledShader, crate::PipelineError> {
    profiling::scope!("compile_dxc");

    let source_name = source_name
        .and_then(|cstr| cstr.to_str().ok())
        .and_then(sanitize_source_name);
    let source_name = source_name.as_deref().map(OPCWSTR::new);
    let raw_ep = OPCWSTR::new(raw_ep);
    let full_stage = OPCWSTR::new(full_stage);

    let debug = device
        .shared
        .private_caps
        .instance_flags
        .contains(wgt::InstanceFlags::DEBUG)
        && !device
            .shared
            .private_caps
            .workarounds
            .avoid_shader_debug_info;
    let enable_16bit = device
        .features
        .intersects(wgt::Features::SHADER_F16 | wgt::Features::SHADER_I16);
    let compile_args = dxc_arguments(
        source_name.as_ref(),
        &raw_ep,
        &full_stage,
        debug,
        enable_16bit,
        denormal_mode,
    );

    let buffer = Dxc::DxcBuffer {
        Ptr: source.as_ptr().cast(),
        Size: source.len(),
        Encoding: Dxc::DXC_CP_UTF8.0,
    };

    let compile_res: Dxc::IDxcResult =
        unsafe { compiler.Compile(&buffer, Some(&compile_args), None) }
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

    let bytecode = unsafe {
        core::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    Ok(bytecode.into())
}

/// Prepare a shader module label for use as DXC's input file name.
///
/// `compile_dxc` passes this as the first positional argument, which DXC reads
/// as the input file's path. Characters that it would misread there are mapped
/// to `-`: control characters and exotic whitespace, path separators, and
/// colons, which DXC takes for a drive designator when the name has no
/// separator.
///
/// The result is anchored to the current directory, so that a label starting
/// with `-` cannot be parsed as a command-line flag. An empty label would leave
/// a bare `./`, which would error since it is a directory, so we drop those.
fn sanitize_source_name(name: &str) -> Option<String> {
    if name.is_empty() {
        return None;
    }

    let name: String = name
        .chars()
        .map(|c| {
            let unsafe_char = c.is_control()
                || std::path::is_separator(c)
                || c == ':'
                || (c.is_whitespace() && c != ' ');
            if unsafe_char {
                '-'
            } else {
                c
            }
        })
        .collect();

    Some(format!("./{name}"))
}

#[cfg(test)]
mod tests {
    use super::sanitize_source_name;

    #[test]
    fn sanitize_source_name_anchors_to_current_dir() {
        assert_eq!(sanitize_source_name("shader").as_deref(), Some("./shader"));
        assert_eq!(
            sanitize_source_name("shader with spaces").as_deref(),
            Some("./shader with spaces")
        );
    }

    #[test]
    fn sanitize_source_name_replaces_path_syntax() {
        assert_eq!(
            sanitize_source_name("Renderer::shader").as_deref(),
            Some("./Renderer--shader")
        );
        assert_eq!(
            sanitize_source_name("renderer/shader").as_deref(),
            Some("./renderer-shader")
        );
        assert_eq!(
            sanitize_source_name("renderer\\shader").as_deref(),
            Some("./renderer-shader")
        );
        assert_eq!(
            sanitize_source_name("C:/shader").as_deref(),
            Some("./C--shader")
        );
    }

    #[test]
    fn sanitize_source_name_defuses_command_line_flags() {
        assert_eq!(sanitize_source_name("-foo").as_deref(), Some("./-foo"));
        assert_eq!(sanitize_source_name("/foo").as_deref(), Some("./-foo"));
    }

    #[test]
    fn sanitize_source_name_replaces_control_chars() {
        assert_eq!(
            sanitize_source_name("shader\twith\ttabs").as_deref(),
            Some("./shader-with-tabs")
        );
        assert_eq!(sanitize_source_name("a\nb").as_deref(), Some("./a-b"));
        assert_eq!(sanitize_source_name("a\u{7}b").as_deref(), Some("./a-b"));
    }

    #[test]
    fn sanitize_source_name_drops_empty() {
        assert_eq!(sanitize_source_name(""), None);
    }
}

#[cfg(test)]
mod denormal_mode_tests {
    use super::*;
    use naga::back::hlsl::ShaderModel;
    use wgt::DxcDenormalMode::{Any, FlushToZero, Preserve};

    #[test]
    fn denormal_arguments_preserve_defaults_and_optional_flags() {
        let name = OPCWSTR::new("./shader");
        let entry = OPCWSTR::new("main");
        let profile = OPCWSTR::new("cs_6_2");
        for named in [false, true] {
            for debug in [false, true] {
                for enable_16bit in [false, true] {
                    let mut expected = Vec::new();
                    if named {
                        expected.push("./shader");
                    }
                    expected.extend([
                        "-E",
                        "main",
                        "-T",
                        "cs_6_2",
                        "-HV",
                        "2018",
                        "-no-warnings",
                        "-Ges",
                    ]);
                    if debug {
                        expected.extend(["-Zi", "-Od"]);
                    }
                    if enable_16bit {
                        expected.push("-enable-16bit-types");
                    }
                    for (mode, value) in [
                        (None, None),
                        (Some(Any), Some("any")),
                        (Some(Preserve), Some("preserve")),
                        (Some(FlushToZero), Some("ftz")),
                    ] {
                        let args = dxc_arguments(
                            named.then_some(&name),
                            &entry,
                            &profile,
                            debug,
                            enable_16bit,
                            mode,
                        );
                        // All backing strings remain in scope during this inspection and Compile.
                        let actual: Vec<_> = args
                            .iter()
                            .map(|arg| unsafe { arg.to_string().unwrap() })
                            .collect();
                        let mut expected = expected.clone();
                        if let Some(value) = value {
                            expected.extend(["-denorm", value]);
                        }
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn denormal_profile_requirements() {
        for profile in [
            ShaderModel::V5_1,
            ShaderModel::V6_0,
            ShaderModel::V6_1,
            ShaderModel::V6_2,
            ShaderModel::V6_7,
            ShaderModel::V6_8,
        ] {
            assert!(validate_denormal_mode(None, profile).is_ok());
            for mode in [Any, Preserve, FlushToZero] {
                assert_eq!(
                    validate_denormal_mode(Some(mode), profile).is_ok(),
                    profile >= ShaderModel::V6_2
                );
            }
        }
    }

    #[test]
    fn denormal_cache_identity() {
        use hashbrown::HashSet;
        let mut keys = HashSet::new();
        for mode in [None, Some(Any), Some(Preserve), Some(FlushToZero)] {
            let key = || super::super::ShaderCacheKey {
                source: String::from("same HLSL"),
                entry_point: String::from("main"),
                stage: naga::ShaderStage::Compute,
                shader_model: ShaderModel::V6_2,
                denormal_mode: mode,
            };
            assert!(keys.insert(key()));
            assert!(!keys.insert(key()));
        }
        assert_eq!(keys.len(), 4);
    }

    #[test]
    fn denormal_fxc_rejects_explicit_modes() {
        // Compiler library only: no D3D12 device or adapter is created.
        assert!(CompilerContainer::new_fxc()
            .unwrap()
            .with_denormal_mode(None)
            .is_ok());
        for mode in [Any, Preserve, FlushToZero] {
            assert!(CompilerContainer::new_fxc()
                .unwrap()
                .with_denormal_mode(Some(mode))
                .is_err());
        }
    }

    #[cfg(static_dxc)]
    #[test]
    fn denormal_static_compiler_selection() {
        for mode in [None, Some(Any), Some(Preserve), Some(FlushToZero)] {
            let container = CompilerContainer::new_static_dxc()
                .unwrap()
                .with_denormal_mode(mode)
                .unwrap();
            assert_eq!(container.denormal_mode(), mode);
        }
    }

    #[test]
    fn denormal_missing_dynamic_compiler() {
        assert!(matches!(
            CompilerContainer::new_dynamic_dxc("nonexistent-wgpu-denormal-test.dll".into()),
            Err(GetContainerError::FailedToLoad(..))
        ));
    }
}

#[cfg(test)]
mod denormal_compiler_tests {
    use super::*;

    // Run the test executable from a directory containing the desired DXC DLL.
    // This uses compiler COM APIs only; it never creates a graphics device.
    #[test]
    #[ignore = "requires dxcompiler.dll in the working directory"]
    fn denormal_actual_dxc() {
        let source = r#"
[noinline] float controlled(float x, float y) { precise float p = x * y; return p; }
float4 main(float4 x : TEXCOORD0) : SV_Target {
    return float4(controlled(x.x, x.y), x.z + x.w, 0.0, 1.0);
}
"#;
        let entry = OPCWSTR::new("main");
        let profile = OPCWSTR::new("ps_6_2");
        for (mode, value) in [
            (None, None),
            (Some(wgt::DxcDenormalMode::Any), Some("any")),
            (Some(wgt::DxcDenormalMode::Preserve), Some("preserve")),
            (Some(wgt::DxcDenormalMode::FlushToZero), Some("ftz")),
        ] {
            let container = CompilerContainer::new_dynamic_dxc(
                std::env::current_dir().unwrap().join("dxcompiler.dll"),
            )
            .unwrap()
            .with_denormal_mode(mode)
            .unwrap();
            assert_eq!(container.denormal_mode(), mode);
            let CompilerContainer::DynamicDxc(ref dxc) = container else {
                unreachable!()
            };
            let args = dxc_arguments(
                None,
                &entry,
                &profile,
                false,
                false,
                container.denormal_mode(),
            );
            let buffer = Dxc::DxcBuffer {
                Ptr: source.as_ptr().cast(),
                Size: source.len(),
                Encoding: Dxc::DXC_CP_UTF8.0,
            };
            let result: Dxc::IDxcResult =
                unsafe { dxc.compiler.Compile(&buffer, Some(&args), None) }.unwrap();
            let errors = get_output::<Dxc::IDxcBlobUtf8>(&result, Dxc::DXC_OUT_ERRORS).unwrap();
            assert!(
                as_err_str(&errors).unwrap().is_empty(),
                "{}",
                as_err_str(&errors).unwrap()
            );
            let object = get_output::<Dxc::IDxcBlob>(&result, Dxc::DXC_OUT_OBJECT).unwrap();
            let buffer = Dxc::DxcBuffer {
                Ptr: unsafe { object.GetBufferPointer() },
                Size: unsafe { object.GetBufferSize() },
                Encoding: 0,
            };
            let disassembly: Dxc::IDxcResult =
                unsafe { dxc.compiler.Disassemble(&buffer) }.unwrap();
            let text =
                get_output::<Dxc::IDxcBlobUtf8>(&disassembly, Dxc::DXC_OUT_DISASSEMBLY).unwrap();
            let text = as_err_str(&text).unwrap();
            std::println!("MODE {mode:?}\n{text}");
            let mut functions = 0;
            for line in text.lines().filter(|line| line.starts_with("define ")) {
                functions += 1;
                let attrs = if let Some((_, tail)) = line.rsplit_once('#') {
                    let group = tail.split_whitespace().next().unwrap();
                    let prefix = format!("attributes #{group} =");
                    text.lines().find(|line| line.starts_with(&prefix)).unwrap()
                } else {
                    ""
                };
                if let Some(value) = value {
                    assert!(attrs.contains(&format!("\"fp32-denorm-mode\"=\"{value}\"")));
                } else {
                    assert!(!attrs.contains("fp32-denorm-mode"));
                }
            }
            assert_eq!(functions, 2);
            assert!(text.contains("fmul float"));
            assert!(!text.contains("fmul fast"));
            assert!(text.contains("fadd fast"));
        }
    }
}
