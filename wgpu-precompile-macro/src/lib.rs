use hashbrown::HashSet;
use proc_macro::TokenStream;
use quote::quote;
use std::path::PathBuf;
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

struct MacroArgs {
    wgpu_crate: Path,
    source_type: SourceType,
    shader_stage: Option<naga::ShaderStage>,
    source: ShaderSource,
    targets: HashSet<CompileTarget>,
    entry_point: String,
    file_name: Option<String>,
}
impl Parse for MacroArgs {
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
impl MacroArgs {
    fn target_enabled(&self, target: CompileTarget) -> bool {
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

/// This is to be re-exported by wgpu in a certain way, so that it can refer to items in the `wgpu` crate
///
/// Input format:
/// precompile!(wgpu_crate_name source_type <shader_stage if glsl input> is_file_path source_string entry_point  targets...)
#[proc_macro]
pub fn precompile(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as MacroArgs);
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

    #[cfg(feature = "spv")]
    let spirv_tokens = if args.target_enabled(CompileTarget::Spirv) {
        let spirv_data = naga::back::spv::write_vec(
            &module,
            &module_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage,
                entry_point: args.entry_point.clone(),
            }),
        )
        .expect("Naga failed to write SPIR-V code");
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::SpirvPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(&[#(#spirv_data),*]),
            })
        }
    } else {
        none_tokens.clone()
    };
    #[cfg(not(feature = "spv"))]
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
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::MslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#msl_str),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#entry_point),
            })
        }
    } else {
        none_tokens.clone()
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
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::HlslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#hlsl_str),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#hlsl_entry_point),
            })
        }
    } else {
        none_tokens.clone()
    };
    #[cfg(not(feature = "hlsl"))]
    let hlsl_tokens = none_tokens.clone();

    #[cfg(feature = "hlsl")]
    let dxil_tokens = if args.target_enabled(CompileTarget::Dxil) {
        todo!()
    } else {
        none_tokens.clone()
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
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::WgslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#wgsl_str),
                entry_point: #wgpu_path::__macro_helpers::ToString::to_string(#entry_point),
            })
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
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
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::GlslPassthroughDescriptor {
                code: #wgpu_path::__macro_helpers::Cow::Borrowed(#glsl_str),
            })
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
        }
    };
    #[cfg(not(feature = "glsl"))]
    let glsl_tokens = none_tokens.clone();

    let label_tokens = match args.file_name {
        Some(f) => quote! {#wgpu_path::__macro_helpers::Some(#f)},
        None => quote! {#wgpu_path::__macro_helpers::None},
    };

    quote! {
        #wgpu_path::ShaderModuleDescriptorPassthrough {
            // TODO: make this something else when file name is provided
            label: #label_tokens,
            num_workgroups: (#x, #y, #z),
            runtime_checks: #wgpu_path::ShaderRuntimeChecks::default(),
            spirv: #spirv_tokens,
            dxil: #dxil_tokens,
            msl: #msl_tokens,
            hlsl: #hlsl_tokens,
            glsl: #glsl_tokens,
            wgsl: #wgsl_tokens,
        }
    }
    .into()
}
