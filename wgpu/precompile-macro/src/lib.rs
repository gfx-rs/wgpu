use std::path::PathBuf;

use hashbrown::HashSet;

use proc_macro::TokenStream;
use syn::{parse::Parse, parse_macro_input, Ident, Path};

enum SourceType {
    Wgsl,
    Glsl,
    Spirv,
}
impl SourceType {
    fn parse(str: &str) -> Option<Self> {
        Some(match str {
            "wgsl" => Self::Wgsl,
            "glsl" => Self::Glsl,
            "spirv" => Self::Spirv,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SourceTarget {
    Wgsl,
    Msl,
    Hlsl,
    Spirv,
    Glsl,
    Dxil,
}
impl SourceTarget {
    fn parse(str: &str) -> Option<Self> {
        Some(match str {
            "wgsl" => Self::Wgsl,
            "msl" => Self::Msl,
            "hlsl" => Self::Hlsl,
            "spirv" => Self::Spirv,
            "glsl" => Self::Glsl,
            "dxil" => Self::Dxil,
            _ => return None,
        })
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
    targets: HashSet<SourceTarget>,
}
impl Parse for MacroArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let wgpu_crate: Path = input.parse()?;
        let source_type =
            SourceType::parse(&input.parse::<Ident>()?.to_string()).expect("Invalid source type");
        let is_file_path = input.parse::<syn::LitBool>()?.value;
        let source_literal = input.parse::<syn::LitStr>()?.value();

        let mut targets = HashSet::new();
        while !input.is_empty() {
            let target = SourceTarget::parse(&input.parse::<syn::Ident>()?.to_string())
                .expect("Invalid target type");
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
            targets,
        })
    }
}

/// This is to be re-exported by wgpu in a certain way, so that it can refer to items in the `wgpu` crate
///
/// Input format:
/// precompile!(wgpu_crate_name, source_type, is_file_path, source_string, targets...)
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
            naga::front::wgsl::parse_str(&source).expect("Nagaဠfailed to parse WGSL input")
        }
        SourceType::Glsl => {
            todo!()
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

    todo!()
}
