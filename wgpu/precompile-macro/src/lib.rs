use hashbrown::HashSet;
use proc_macro::TokenStream;
use quote::quote;
use std::path::PathBuf;
use syn::{parse::Parse, parse_macro_input, Ident, Path};

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
            other => panic!("Unrecognized compile target: {other}"),
        }
    }
}

fn parse_shader_stage(str: &str) -> naga::ShaderStage {
    match str {
        "vertex" => naga::ShaderStage::Vertex,
        "fragment" => naga::ShaderStage::Fragment,
        "compute" => naga::ShaderStage::Compute,
        "Task" => naga::ShaderStage::Task,
        "Mesh" => naga::ShaderStage::Mesh,
        other => panic!("Unrecognized shader stage: {other}"),
    }
}

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

struct MacroArgs {
    wgpu_crate: Path,
    source_type: SourceType,
    source: ShaderSource,
    targets: HashSet<CompileTarget>,
    entry_point: String,
    shader_stage: naga::ShaderStage,
}
impl Parse for MacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let wgpu_crate: Path = input.parse()?;
        let source_type = SourceType::parse(&input.parse::<Ident>()?.to_string());
        let is_file_path = input.parse::<syn::LitBool>()?.value;
        let source_literal = input.parse::<syn::LitStr>()?.value();
        let entry_point = input.parse::<syn::LitStr>()?.value();
        let shader_stage = parse_shader_stage(&input.parse::<Ident>()?.to_string());

        let mut targets = HashSet::new();
        while !input.is_empty() {
            let target = CompileTarget::parse(&input.parse::<syn::Ident>()?.to_string());
            targets.insert(target);
        }

        let source = if is_file_path {
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let relative_path = PathBuf::from(source_literal);
            let path_to_read = if relative_path.is_relative() {
                PathBuf::from(manifest_dir).join(relative_path)
            } else {
                relative_path
            };
            let bytes = std::fs::read(path_to_read).expect("Failed to read input file");
            ShaderSource::File(bytes)
        } else {
            ShaderSource::String(source_literal)
        };
        Ok(Self {
            wgpu_crate,
            source_type,
            source,
            entry_point,
            shader_stage,
            targets,
        })
    }
}

/// This is to be re-exported by wgpu in a certain way, so that it can refer to items in the `wgpu` crate
///
/// Input format:
/// precompile!(wgpu_crate_name source_type is_file_path source_string entry_point shader_stage targets...)
#[proc_macro]
pub fn precompile(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as MacroArgs);
    let module = match args.source_type {
        SourceType::Spirv => {
            let source = args.source.expect_bytes();
            let options = naga::front::spv::Options {
                adjust_coordinate_space: false, // we require NDC_Y_UP feature
                strict_capabilities: true,
                block_ctx_dump_prefix: None,
            };
            naga::front::spv::parse_u8_slice(&source, &options)
                .expect("Naga failed to parse SPIR-V input")
        }
        SourceType::Wgsl => {
            let source = args.source.expect_string();
            naga::front::wgsl::parse_str(&source).expect("Naga failed to parse WGSL input")
        }
        SourceType::Glsl => {
            panic!("GLSL parsing support is not yet available")
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

    let entry_point_idx = module
        .entry_points
        .iter()
        .position(|a| a.name == args.entry_point)
        .expect("Requested entry point not present in module");

    if args.shader_stage != module.entry_points[entry_point_idx].stage {
        panic!(
            "Incorrect shader stage: given {:?} but entry point has stage {:?}",
            args.shader_stage, module.entry_points[entry_point_idx].stage
        )
    }

    let wgpu_path = &args.wgpu_crate;

    let spirv_tokens = if args.targets.contains(&CompileTarget::Spirv) {
        let spirv_data = naga::back::spv::write_vec(
            &module,
            &module_info,
            &naga::back::spv::Options::default(),
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: args.shader_stage,
                entry_point: args.entry_point.clone(),
            }),
        )
        .expect("Naga failed to write SPIR-V code");
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::__macro_helpers::Cow::Borrowed(&[#(#spirv_data),*]))
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
        }
    };

    let msl_tokens = if args.targets.contains(&CompileTarget::Msl) {
        let msl_str = naga::back::msl::write_string(
            &module,
            &module_info,
            &naga::back::msl::Options::default(),
            &naga::back::msl::PipelineOptions {
                entry_point: Some((args.shader_stage, args.entry_point.clone())),
                ..Default::default()
            },
        )
        .expect("Naga failed to write MSL code")
        .0;
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::__macro_helpers::Cow::Borrowed(#msl_str))
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
        }
    };

    let hlsl_tokens = if args.targets.contains(&CompileTarget::Hlsl) {
        let mut hlsl_str = String::new();
        naga::back::hlsl::Writer::new(
            &mut hlsl_str,
            &naga::back::hlsl::Options::default(),
            &naga::back::hlsl::PipelineOptions {
                entry_point: Some((args.shader_stage, args.entry_point.clone())),
            },
        )
        .write(&module, &module_info, None)
        .expect("Naga failed to write HLSL code");
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::__macro_helpers::Cow::Borrowed(#hlsl_str))
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
        }
    };

    let wgsl_tokens = if args.targets.contains(&CompileTarget::Wgsl) {
        let mut wgsl_str = String::new();
        naga::back::wgsl::Writer::new(&mut wgsl_str, naga::back::wgsl::WriterFlags::empty())
            .write(&module, &module_info)
            .expect("Naga failed to write WGSL code");
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::__macro_helpers::Cow::Borrowed(#wgsl_str))
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
        }
    };

    let glsl_tokens = if args.targets.contains(&CompileTarget::Glsl) {
        let mut glsl_str = String::new();
        naga::back::glsl::Writer::new(
            &mut glsl_str,
            &module,
            &module_info,
            &naga::back::glsl::Options::default(),
            &naga::back::glsl::PipelineOptions {
                shader_stage: args.shader_stage,
                entry_point: args.entry_point.clone(),
                multiview: None,
            },
            naga::proc::BoundsCheckPolicies::default(),
        )
        .expect("Naga failed to create GLSL writer")
        .write()
        .expect("Naga failed write GLSL code");
        quote! {
            #wgpu_path::__macro_helpers::Some(#wgpu_path::__macro_helpers::Cow::Borrowed(#glsl_str))
        }
    } else {
        quote! {
            #wgpu_path::__macro_helpers::None
        }
    };

    let [x, y, z] = module.entry_points[entry_point_idx].workgroup_size;
    let entry_point = &args.entry_point;

    quote! {
        #wgpu_path::ShaderModuleDescriptorPassthrough {
            // TODO: make this something else when file name is provided
            label: #wgpu_path::__macro_helpers::None,
            entry_point: #wgpu_path::__macro_helpers::String::from(#entry_point),
            num_workgroups: (#x, #y, #z),
            runtime_checks: #wgpu_path::ShaderRuntimeChecks::unchecked(),
            spirv: #spirv_tokens,
            dxil: #wgpu_path::__macro_helpers::None,
            msl: #msl_tokens,
            hlsl: #hlsl_tokens,
            glsl: #glsl_tokens,
            wgsl: #wgsl_tokens,
        }
    }
    .into()
}
