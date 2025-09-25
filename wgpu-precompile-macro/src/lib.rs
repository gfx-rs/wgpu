use hashbrown::HashSet;
use proc_macro::TokenStream;
use quote::quote;
use std::{path::PathBuf, process::Stdio};
use syn::{parse::Parse, parse_macro_input, Ident, Path};

#[derive(PartialEq, Eq)]
enum SourceType {
    Wgsl,
    Glsl,
    Spirv,
}
impl SourceType {
    fn parse(str: &str) -> Self {
        match str {
            "wgsl" => Self::Wgsl,
            "glsl" => Self::Glsl,
            "spirv" => Self::Spirv,
            other => panic!("Unrecognized source type: {other}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CompileTarget {
    Wgsl,
    Msl,
    Hlsl,
    Spirv,
    Glsl,
    Dxil,
    AllSupported,
}
impl CompileTarget {
    fn parse(str: &str) -> Self {
        match str {
            "wgsl" => Self::Wgsl,
            "msl" => Self::Msl,
            "hlsl" => Self::Hlsl,
            "spirv" => Self::Spirv,
            "glsl" => Self::Glsl,
            "dxil" => Self::Dxil,
            "all" => Self::AllSupported,
            other => panic!("Unrecognized compile target: {other}"),
        }
    }
}

#[derive(Clone)]
enum ShaderSource {
    String(String),
    File(Vec<u8>),
}
impl ShaderSource {
    fn expect_bytes(self) -> Vec<u8> {
        match self {
            Self::File(f) => f,
            Self::String(_) => panic!("Spirv input must be a file path"),
        }
    }
    fn expect_string(self) -> String {
        match self {
            Self::String(s) => s,
            Self::File(file) => String::from_utf8(file)
                .expect("Expected string input, but file couldn't be parsed as UTF-8"),
        }
    }
}

fn parse_shader_stage(str: &str) -> Option<naga::ShaderStage> {
    match str {
        "vertex" => Some(naga::ShaderStage::Vertex),
        "fragment" => Some(naga::ShaderStage::Fragment),
        "compute" => Some(naga::ShaderStage::Compute),
        "task" => Some(naga::ShaderStage::Task),
        "mesh" => Some(naga::ShaderStage::Mesh),
        _ => None,
    }
}

fn shader_stage_to_string(stage: naga::ShaderStage) -> &'static str {
    match stage {
        naga::ShaderStage::Vertex => "vertex",
        naga::ShaderStage::Fragment => "fragment",
        naga::ShaderStage::Compute => "compute",
        naga::ShaderStage::Mesh => "mesh",
        naga::ShaderStage::Task => "task",
    }
}

struct PrecompileArgs {
    wgpu_crate: Path,
    source_type: SourceType,
    shader_stage: Option<naga::ShaderStage>,
    source: ShaderSource,
    targets: HashSet<CompileTarget>,
    entry_point: String,
    file_name: Option<String>,
}
impl Parse for PrecompileArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let wgpu_crate: Path = input.parse()?;
        let source_type = SourceType::parse(&input.parse::<Ident>()?.to_string());
        let shader_stage = if source_type == SourceType::Glsl {
            let ident = input.parse::<Ident>()?.to_string();
            let stage = parse_shader_stage(&ident);
            if stage.is_none() {
                panic!("Invalid shader stage for GLSL: {ident}");
            }
            stage
        } else {
            None
        };
        let is_file_path = input.parse::<syn::LitBool>()?.value;
        let source_literal = input.parse::<syn::LitStr>()?.value();
        let entry_point = input.parse::<syn::LitStr>()?.value();

        let mut targets = HashSet::new();
        while !input.is_empty() {
            let target = CompileTarget::parse(&input.parse::<syn::Ident>()?.to_string());
            targets.insert(target);
        }

        let (source, file_name) = if is_file_path {
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let relative_path = PathBuf::from(source_literal);
            let file_name = relative_path
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned();
            let path_to_read = if relative_path.is_relative() {
                PathBuf::from(manifest_dir).join(relative_path)
            } else {
                relative_path
            };
            if !path_to_read.is_file() {
                panic!("Path does not exist or is not a file: {path_to_read:?}")
            }
            let bytes = std::fs::read(path_to_read).expect("Failed to read input file");
            (ShaderSource::File(bytes), Some(file_name))
        } else {
            (ShaderSource::String(source_literal), None)
        };
        Ok(Self {
            wgpu_crate,
            source_type,
            shader_stage,
            source,
            entry_point,
            targets,
            file_name,
        })
    }
}
impl PrecompileArgs {
    fn target_enabled(&self, target: CompileTarget) -> bool {
        // TODO: only enable if we are actually targeting that platform.
        // This is especially important for DXIL, which requires dxc. No need
        // to fail to compile on MacOS if dxc isn't present!
        match target {
            CompileTarget::Dxil => self.targets.contains(&CompileTarget::Dxil),
            CompileTarget::Hlsl => {
                (self.targets.contains(&CompileTarget::Hlsl)
                    || self.targets.contains(&CompileTarget::AllSupported))
                    && !self.targets.contains(&CompileTarget::Dxil)
            }
            CompileTarget::AllSupported => unreachable!(),
            other => {
                self.targets.contains(&other) || self.targets.contains(&CompileTarget::AllSupported)
            }
        }
    }
}

/// All fields represented as string literals
struct PrecompileDxilArgs {
    hlsl_code: String,
    entry_point: String,
    shader_stage: naga::ShaderStage,
}
impl Parse for PrecompileDxilArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let hlsl_code = input.parse::<syn::LitStr>()?.value();
        let entry_point = input.parse::<syn::LitStr>()?.value();
        let shader_stage = parse_shader_stage(&input.parse::<syn::LitStr>()?.value()).unwrap();
        Ok(Self {
            hlsl_code,
            entry_point,
            shader_stage,
        })
    }
}

struct TempFolder {
    path: PathBuf,
}
impl Drop for TempFolder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// This is so we can conditionally hook into DXC depending on the target, which must be configured via #\[cfg] within the main program.
/// Proc macros don't directly have access to the target configuration. Invoking naga unnecessarily shouldn't be *too* major, but
/// compiling to macOS shouldn't require dxc be present.
#[proc_macro]
pub fn precompile_hlsl_to_dxil(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as PrecompileDxilArgs);
    let target_profile = match args.shader_stage {
        naga::ShaderStage::Vertex => "vs_5_1",
        naga::ShaderStage::Fragment => "ps_5_1",
        naga::ShaderStage::Compute => "cs_5_1",
        naga::ShaderStage::Task => "as_5_1",
        naga::ShaderStage::Mesh => "ms_5_1",
    };

    let temporary_folder_location = std::env::temp_dir().join(
        getrandom::u64()
            .expect("Failed to generate random u64")
            .to_string(),
    );
    std::fs::create_dir(&temporary_folder_location).expect("Failed to create temporary directory");
    // Drop guard essentially
    let tempoary_folder = TempFolder {
        path: temporary_folder_location,
    };

    // The naming matters for DXIL debug info. We don't want to give it a name that seems like something the
    // user might've specified, as that could cause confusion.
    let input_file = tempoary_folder.path.join("__wgpu_inline.hlsl");
    std::fs::write(&input_file, args.hlsl_code.as_bytes())
        .expect("Failed to write to HLSL input file");
    let temporary_file_location = tempoary_folder.path.join("file.dxil");
    let output = std::process::Command::new("dxc")
        .args([
            "-T",
            target_profile,
            "-E",
            &args.entry_point,
            &input_file.display().to_string(),
            "-Fo",
            &temporary_file_location.display().to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .output()
        .expect("Failed to spawn DXC");
    if !output.status.success() {
        panic!("DXC failed:\n{}", String::from_utf8(output.stderr).unwrap());
    }
    let dxil = std::fs::read(temporary_file_location).expect("Failed to read DXC output file");
    quote! {
        &[#(#dxil),*]
    }
    .into()
}

fn generate_conditional_guard(target: CompileTarget) -> proc_macro2::TokenStream {
    // This is for my sanity. The guards should always be used within conditional code anyway.
    #[allow(unused_variables)]
    let always_false = quote! {
        any()
    };
    // This basically mirrors the things in wgpu-core/platform-deps
    match target {
        CompileTarget::Msl => {
            // Related features: metal
            #[cfg(feature = "metal")]
            quote! {
                target_vendor = "apple"
            }
            // Always false
            #[cfg(not(feature = "metal"))]
            always_false
        }
        CompileTarget::Glsl => {
            // Related features: gles, webgl, angle
            let webgl = if cfg!(feature = "webgl") {
                quote! {
                    ,all(target_arch = "wasm32", not(target_os = "emscripten"))
                }
            } else {
                quote! {}
            };
            let angle = if cfg!(feature = "angle") {
                quote! {
                    ,target_vendor = "apple"
                }
            } else {
                quote! {}
            };
            #[cfg(feature = "gles")]
            quote! {
                any(target_os = "emscripten", windows, target_os = "linux", target_os = "android", target_os = "freebsd" #angle #webgl)
            }
            #[cfg(not(feature = "gles"))]
            always_false
        }
        CompileTarget::Spirv => {
            // Related features: vulkan, vulkan-portability
            #[cfg(all(feature = "vulkan", feature = "vulkan-portability"))]
            quote! {
                any(target_vendor = "apple", windows, target_os = "linux", target_os = "android", target_os = "freebsd")
            }
            #[cfg(all(feature = "vulkan", not(feature = "vulkan-portability")))]
            quote! {
                any(windows, target_os = "linux", target_os = "android", target_os = "freebsd")
            }
            #[cfg(not(feature = "vulkan"))]
            always_false
        }
        CompileTarget::Wgsl => {
            // Related features: webgpu
            #[cfg(feature = "webgpu")]
            quote! {
                all(target_arch = "wasm32", not(target_os = "emscripten"))
            }
            #[cfg(not(feature = "webgpu"))]
            always_false
        }
        CompileTarget::Hlsl | CompileTarget::Dxil => {
            // Related features: dx12
            // Note: this differs slightly from platform-deps which enables dx12 even on linux/android, but just doesn't
            // let you use it
            #[cfg(feature = "dx12")]
            quote! {
                windows
            }
            #[cfg(not(feature = "dx12"))]
            always_false
        }
        CompileTarget::AllSupported => unreachable!(),
    }
}

/// This is to be re-exported by wgpu in a certain way, so that it can refer to items in the `wgpu` crate
///
/// Input format:
/// precompile!(wgpu_crate_name source_type <shader_stage if glsl input> is_file_path source_string entry_point  targets...)
#[proc_macro]
pub fn precompile(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as PrecompileArgs);
    let module = match args.source_type {
        SourceType::Spirv => {
            let source = args.source.clone().expect_bytes();
            // This is yanked from wgpu-hal. Maybe at some point this kinda logic should be unified somewhere
            let options = naga::front::spv::Options {
                adjust_coordinate_space: false, // we require NDC_Y_UP feature
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            };
            naga::front::spv::parse_u8_slice(&source, &options)
                .expect("Naga failed to parse SPIR-V input")
        }
        SourceType::Wgsl => {
            let source = args.source.clone().expect_string();
            naga::front::wgsl::Frontend::new_with_options(naga::front::wgsl::Options {
                parse_doc_comments: false,
            })
            .parse(&source)
            .expect("Naga failed to parse WGSL input")
        }
        SourceType::Glsl => {
            let src = args.source.clone().expect_string();
            naga::front::glsl::Frontend::default()
                .parse(
                    &naga::front::glsl::Options {
                        // This is guaranteed to be some
                        stage: args.shader_stage.unwrap(),
                        defines: Default::default(),
                    },
                    &src,
                )
                .expect("Naga failed to parse GLSL input")
        }
    };

    let module_info: naga::valid::ModuleInfo = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .subgroup_stages(naga::valid::ShaderStages::all())
    .subgroup_operations(naga::valid::SubgroupOperationSet::all())
    .validate(&module)
    .expect("Naga failed to validate module");

    let entry_point = module
        .entry_points
        .iter()
        .find(|a| a.name == args.entry_point)
        .expect("Requested entry point not present in module");
    let shader_stage = entry_point.stage;
    let [x, y, z] = entry_point.workgroup_size;

    let wgpu_path = &args.wgpu_crate;

    let none_tokens = quote! {
        #wgpu_path::__macro_helpers::None
    };

    #[cfg(feature = "spirv")]
    let spirv_tokens = if args.target_enabled(CompileTarget::Spirv) {
        use naga::back::spv;
        // Ripped from wgpu-hal backend
        let mut flags = spv::WriterFlags::empty();
        flags.set(spv::WriterFlags::DEBUG, false);
        flags.set(spv::WriterFlags::LABEL_VARYINGS, false);
        flags.set(
            spv::WriterFlags::FORCE_POINT_SIZE,
            //Note: we could technically disable this when we are compiling separate entry points,
            // and we know exactly that the primitive topology is not `PointList`.
            // But this requires cloning the `spv::Options` struct, which has heap allocations.
            true, // could check `super::Workarounds::SEPARATE_ENTRY_POINTS`
        );
        let spirv_data = spv::write_vec(
            &module,
            &module_info,
            &naga::back::spv::Options {
                flags,
                ..Default::default()
            },
            Some(&naga::back::spv::PipelineOptions {
                shader_stage,
                entry_point: args.entry_point.clone(),
            }),
        )
        .expect("Naga failed to write SPIR-V code");
        let guard = generate_conditional_guard(CompileTarget::Spirv);
        quote! {
            #[cfg(#guard)]
            spirv: #wgpu_path::__macro_helpers::Some(#wgpu_path::SpirvPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(&[#(#spirv_data),*]),
            }),
            #[cfg(not(#guard))]
            spirv: #none_tokens,
        }
    } else {
        quote! {
            spirv: #none_tokens,
        }
    };
    #[cfg(not(feature = "spirv"))]
    let spirv_tokens = none_tokens.clone();

    #[cfg(feature = "msl")]
    let msl_tokens = if args.target_enabled(CompileTarget::Msl) {
        let (msl_str, translation_info) = naga::back::msl::write_string(
            &module,
            &module_info,
            &naga::back::msl::Options::default(),
            &naga::back::msl::PipelineOptions {
                entry_point: Some((shader_stage, args.entry_point.clone())),
                ..Default::default()
            },
        )
        .expect("Naga failed to write MSL code");
        let entry_point = translation_info.entry_point_names[0].as_ref().unwrap();
        let guard = generate_conditional_guard(CompileTarget::Msl);
        quote! {
            #[cfg(#guard)]
            msl: #wgpu_path::__macro_helpers::Some(#wgpu_path::MslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#msl_str),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#entry_point),
            }),
            #[cfg(not(#guard))]
            msl: #none_tokens,
        }
    } else {
        quote! {
            msl: #none_tokens,
        }
    };
    #[cfg(not(feature = "msl"))]
    let msl_tokens = none_tokens.clone();

    #[cfg(feature = "hlsl")]
    let (hlsl_str, hlsl_entry_point) =
        if args.target_enabled(CompileTarget::Hlsl) || args.target_enabled(CompileTarget::Dxil) {
            let mut hlsl_str = String::new();
            let reflection = naga::back::hlsl::Writer::new(
                &mut hlsl_str,
                &naga::back::hlsl::Options::default(),
                &naga::back::hlsl::PipelineOptions {
                    entry_point: Some((shader_stage, args.entry_point.clone())),
                },
            )
            .write(&module, &module_info, None)
            .expect("Naga failed to write HLSL code");
            let entry_point = reflection.entry_point_names[0].as_ref().unwrap();
            (hlsl_str, entry_point.clone())
        } else {
            (String::new(), String::new())
        };

    #[cfg(feature = "hlsl")]
    let hlsl_tokens = if args.target_enabled(CompileTarget::Hlsl) {
        let guard = generate_conditional_guard(CompileTarget::Hlsl);
        quote! {
            #[cfg(#guard)]
            hlsl: #wgpu_path::__macro_helpers::Some(#wgpu_path::HlslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#hlsl_str),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#hlsl_entry_point),
            }),
            #[cfg(not(#guard))]
            hlsl: #none_tokens,
        }
    } else {
        quote! {
            hlsl: #none_tokens,
        }
    };
    #[cfg(not(feature = "hlsl"))]
    let hlsl_tokens = none_tokens.clone();

    #[cfg(feature = "hlsl")]
    let dxil_tokens = if args.target_enabled(CompileTarget::Dxil) {
        let shader_stage = shader_stage_to_string(shader_stage);
        let guard = generate_conditional_guard(CompileTarget::Dxil);
        quote! {
            #[cfg(#guard)]
            dxil: #wgpu_path::__macro_helpers::Some(#wgpu_path::DxilPassthroughDescriptor {
                // HLSL, entry, shader stage, filename
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#wgpu_path::__macro_helpers::precompile_hlsl_to_dxil!(#hlsl_str #hlsl_entry_point #shader_stage)),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#hlsl_entry_point),
            }),
            #[cfg(not(#guard))]
            dxil: #none_tokens,
        }
    } else {
        quote! {
            dxil: #none_tokens,
        }
    };
    #[cfg(not(feature = "hlsl"))]
    let dxil_tokens = none_tokens.clone();

    #[cfg(feature = "wgsl")]
    let wgsl_tokens = if args.target_enabled(CompileTarget::Wgsl) {
        let mut writer =
            naga::back::wgsl::Writer::new(String::new(), naga::back::wgsl::WriterFlags::empty());
        writer
            .write(&module, &module_info)
            .expect("Naga failed to write WGSL code");
        let wgsl_str = writer.finish();
        // TODO: ensure that the entry point here is sensible
        let entry_point = &args.entry_point;
        let guard = generate_conditional_guard(CompileTarget::Wgsl);
        quote! {
            #[cfg(#guard)]
            wgsl: #wgpu_path::__macro_helpers::Some(#wgpu_path::WgslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#wgsl_str),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#entry_point),
            }),
            #[cfg(not(#guard))]
            wgsl: #none_tokens,

        }
    } else {
        quote! {
            wgsl: #none_tokens,
        }
    };
    #[cfg(not(feature = "wgsl"))]
    let wgsl_tokens = none_tokens.clone();

    #[cfg(feature = "glsl")]
    let glsl_tokens = if args.target_enabled(CompileTarget::Glsl) {
        let mut glsl_str = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl_str,
            &module,
            &module_info,
            &naga::back::glsl::Options::default(),
            &naga::back::glsl::PipelineOptions {
                shader_stage,
                entry_point: args.entry_point.clone(),
                multiview: None,
            },
            naga::proc::BoundsCheckPolicies::default(),
        )
        .expect("Naga failed to create GLSL writer")
        .write()
        .expect("Naga failed write GLSL code");
        let guard = generate_conditional_guard(CompileTarget::Glsl);
        quote! {
            #[cfg(#guard)]
            glsl: #wgpu_path::__macro_helpers::Some(#wgpu_path::GlslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#glsl_str),
            }),
            #[cfg(not(#guard))]
            glsl: #none_tokens,
        }
    } else {
        quote! {
            glsl: #none_tokens,
        }
    };
    #[cfg(not(feature = "glsl"))]
    let glsl_tokens = none_tokens.clone();

    let label_tokens = match args.file_name {
        Some(f) => quote! {#wgpu_path::__macro_helpers::Some(#f)},
        None => quote! {#wgpu_path::__macro_helpers::None},
    };

    let f = quote! {
        #wgpu_path::ShaderModuleDescriptorPassthrough {
            // TODO: make this something else when file name is provided
            label: #label_tokens,
            num_workgroups: (#x, #y, #z),
            runtime_checks: #wgpu_path::ShaderRuntimeChecks::default(),
            #spirv_tokens
            #dxil_tokens
            #msl_tokens
            #hlsl_tokens
            #glsl_tokens
            #wgsl_tokens
        }
    };
    //panic!("Final tokenstream: {}", f);
    f.into()
}
