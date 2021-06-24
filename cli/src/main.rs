#![allow(clippy::manual_strip)]
#[allow(unused_imports)]
use std::fs;
use std::{env, error::Error, path::Path};

#[derive(Default)]
struct Parameters {
    validation_flags: naga::valid::ValidationFlags,
    index_bounds_check_policy: naga::back::IndexBoundsCheckPolicy,
    spv_adjust_coordinate_space: bool,
    spv_flow_dump_prefix: Option<String>,
    spv: naga::back::spv::Options,
    msl: naga::back::msl::Options,
    glsl: naga::back::glsl::Options,
    hlsl: naga::back::hlsl::Options,
}

trait PrettyResult {
    type Target;
    fn unwrap_pretty(self) -> Self::Target;
}

fn print_err(error: impl Error) {
    eprint!("{}", error);

    let mut e = error.source();
    if e.is_some() {
        eprintln!(": ");
    } else {
        eprintln!();
    }

    while let Some(source) = e {
        eprintln!("\t{}", source);
        e = source.source();
    }
}

impl<T, E: Error> PrettyResult for Result<T, E> {
    type Target = T;
    fn unwrap_pretty(self) -> T {
        match self {
            Result::Ok(value) => value,
            Result::Err(error) => {
                print_err(error);
                std::process::exit(1);
            }
        }
    }
}

fn main() {
    env_logger::init();

    let mut input_path = None;
    let mut output_paths = Vec::new();
    //TODO: read the parameters from RON?
    #[allow(unused_mut)]
    let mut params = Parameters::default();

    let mut args = env::args();
    let _ = args.next().unwrap();
    #[allow(clippy::while_let_on_iterator)]
    while let Some(arg) = args.next() {
        //TODO: use `strip_prefix` when MSRV reaches 1.45.0
        if arg.starts_with("--") {
            match &arg[2..] {
                "validate" => {
                    let value = args.next().unwrap().parse().unwrap();
                    params.validation_flags =
                        naga::valid::ValidationFlags::from_bits(value).unwrap();
                }
                "index-bounds-check-policy" => {
                    let value = args.next().unwrap();
                    params.index_bounds_check_policy = match value.as_str() {
                        "Restrict" => naga::back::IndexBoundsCheckPolicy::Restrict,
                        "ReadZeroSkipWrite" => {
                            naga::back::IndexBoundsCheckPolicy::ReadZeroSkipWrite
                        }
                        "UndefinedBehavior" => {
                            naga::back::IndexBoundsCheckPolicy::UndefinedBehavior
                        }
                        other => {
                            panic!(
                                "Unrecognized '--index-bounds-check-policy' value: {:?}",
                                other
                            );
                        }
                    };
                }
                "flow-dir" => params.spv_flow_dump_prefix = args.next(),
                "entry-point" => params.glsl.entry_point = args.next().unwrap(),
                "profile" => {
                    use naga::back::glsl::Version;
                    let string = args.next().unwrap();
                    //TODO: use `strip_prefix` in 1.45.0
                    params.glsl.version = if string.starts_with("core") {
                        Version::Desktop(string[4..].parse().unwrap_or(330))
                    } else if string.starts_with("es") {
                        Version::Embedded(string[2..].parse().unwrap_or(310))
                    } else {
                        panic!("Unknown profile: {}", string)
                    };
                }
                "shader-model" => {
                    use naga::back::hlsl::ShaderModel;
                    let string = args.next().unwrap();
                    let sm_numb = string.parse::<u16>().unwrap();
                    let sm = match string.parse().unwrap() {
                        50 => ShaderModel::V5_0,
                        51 => ShaderModel::V5_1,
                        60 => ShaderModel::V6_0,
                        _ => panic!("Unsupported shader model: {}", sm_numb),
                    };
                    params.hlsl.shader_model = sm;
                }
                other => log::warn!("Unknown parameter: {}", other),
            }
        } else if input_path.is_none() {
            input_path = Some(arg);
        } else {
            output_paths.push(arg);
        }
    }

    let input_path = match input_path {
        Some(ref string) => Path::new(string),
        None => {
            println!("Call with <input> <output> [<options>]");
            return;
        }
    };
    let module = match Path::new(input_path)
        .extension()
        .expect("Input has no extension?")
        .to_str()
        .unwrap()
    {
        "spv" => {
            let options = naga::front::spv::Options {
                adjust_coordinate_space: params.spv_adjust_coordinate_space,
                strict_capabilities: false,
                flow_graph_dump_prefix: params.spv_flow_dump_prefix.map(std::path::PathBuf::from),
            };
            let input = fs::read(input_path).unwrap();
            naga::front::spv::parse_u8_slice(&input, &options).unwrap()
        }
        "wgsl" => {
            let input = fs::read_to_string(input_path).unwrap();
            let result = naga::front::wgsl::parse_str(&input);
            match result {
                Ok(v) => v,
                Err(ref e) => {
                    e.emit_to_stderr(&input);
                    panic!("unable to parse WGSL");
                }
            }
        }
        "vert" => {
            let input = fs::read_to_string(input_path).unwrap();
            let mut entry_points = naga::FastHashMap::default();
            entry_points.insert("main".to_string(), naga::ShaderStage::Vertex);
            naga::front::glsl::parse_str(
                &input,
                &naga::front::glsl::Options {
                    entry_points,
                    defines: Default::default(),
                },
            )
            .unwrap_or_else(|err| {
                let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                emit_glsl_parser_error(err, filename.unwrap_or("glsl"), &input);
                std::process::exit(1);
            })
        }
        "frag" => {
            let input = fs::read_to_string(input_path).unwrap();
            let mut entry_points = naga::FastHashMap::default();
            entry_points.insert("main".to_string(), naga::ShaderStage::Fragment);
            naga::front::glsl::parse_str(
                &input,
                &naga::front::glsl::Options {
                    entry_points,
                    defines: Default::default(),
                },
            )
            .unwrap_or_else(|err| {
                let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                emit_glsl_parser_error(err, filename.unwrap_or("glsl"), &input);
                std::process::exit(1);
            })
        }
        "comp" => {
            let input = fs::read_to_string(input_path).unwrap();
            let mut entry_points = naga::FastHashMap::default();
            entry_points.insert("main".to_string(), naga::ShaderStage::Compute);
            naga::front::glsl::parse_str(
                &input,
                &naga::front::glsl::Options {
                    entry_points,
                    defines: Default::default(),
                },
            )
            .unwrap_or_else(|err| {
                let filename = input_path.file_name().and_then(std::ffi::OsStr::to_str);
                emit_glsl_parser_error(err, filename.unwrap_or("glsl"), &input);
                std::process::exit(1);
            })
        }
        other => panic!("Unknown input extension: {}", other),
    };

    // validate the IR
    let info = match naga::valid::Validator::new(
        params.validation_flags,
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    {
        Ok(info) => Some(info),
        Err(error) => {
            print_err(error);
            None
        }
    };

    if output_paths.is_empty() {
        if info.is_some() {
            println!("Validation successful");
            return;
        } else {
            std::process::exit(!0);
        }
    }

    for output_path in output_paths {
        match Path::new(&output_path)
            .extension()
            .expect("Output has no extension?")
            .to_str()
            .unwrap()
        {
            "txt" => {
                use std::io::Write;

                let mut file = fs::File::create(output_path).unwrap();
                writeln!(file, "{:#?}", module).unwrap();
                if let Some(ref info) = info {
                    writeln!(file).unwrap();
                    writeln!(file, "{:#?}", info).unwrap();
                }
            }
            "metal" => {
                use naga::back::msl;

                let pipeline_options = msl::PipelineOptions::default();
                let (msl, _) = msl::write_string(
                    &module,
                    info.as_ref().unwrap(),
                    &params.msl,
                    &pipeline_options,
                )
                .unwrap_pretty();
                fs::write(output_path, msl).unwrap();
            }
            "spv" => {
                use naga::back::spv;

                params.spv.index_bounds_check_policy = params.index_bounds_check_policy;

                let spv =
                    spv::write_vec(&module, info.as_ref().unwrap(), &params.spv).unwrap_pretty();
                let bytes = spv
                    .iter()
                    .fold(Vec::with_capacity(spv.len() * 4), |mut v, w| {
                        v.extend_from_slice(&w.to_le_bytes());
                        v
                    });

                fs::write(output_path, bytes.as_slice()).unwrap();
            }
            stage @ "vert" | stage @ "frag" | stage @ "comp" => {
                use naga::back::glsl;

                params.glsl.shader_stage = match stage {
                    "vert" => naga::ShaderStage::Vertex,
                    "frag" => naga::ShaderStage::Fragment,
                    "comp" => naga::ShaderStage::Compute,
                    _ => unreachable!(),
                };

                let mut buffer = String::new();
                let mut writer =
                    glsl::Writer::new(&mut buffer, &module, info.as_ref().unwrap(), &params.glsl)
                        .unwrap_pretty();
                writer.write().unwrap();
                fs::write(output_path, buffer).unwrap();
            }
            "dot" => {
                use naga::back::dot;

                let output = dot::write(&module, info.as_ref()).unwrap();
                fs::write(output_path, output).unwrap();
            }
            "hlsl" => {
                use naga::back::hlsl;
                let mut buffer = String::new();
                let mut writer = hlsl::Writer::new(&mut buffer, &params.hlsl);
                writer
                    .write(&module, info.as_ref().unwrap())
                    .unwrap_pretty();
                fs::write(output_path, buffer).unwrap();
            }
            "wgsl" => {
                use naga::back::wgsl;

                let wgsl = wgsl::write_string(&module, info.as_ref().unwrap()).unwrap_pretty();
                fs::write(output_path, wgsl).unwrap();
            }
            other => {
                println!("Unknown output extension: {}", other);
            }
        }
    }
}

use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    files::SimpleFile,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};

pub fn emit_glsl_parser_error(err: naga::front::glsl::ParseError, filename: &str, source: &str) {
    let diagnostic = match err.kind.metadata() {
        Some(metadata) => Diagnostic::error()
            .with_message(err.kind.to_string())
            .with_labels(vec![Label::primary((), metadata.start..metadata.end)]),
        None => Diagnostic::error().with_message(err.kind.to_string()),
    };

    let files = SimpleFile::new(filename, source);
    let config = codespan_reporting::term::Config::default();
    let writer = StandardStream::stderr(ColorChoice::Auto);
    term::emit(&mut writer.lock(), &config, &files, &diagnostic).expect("cannot write error");
}
