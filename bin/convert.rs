#![allow(clippy::manual_strip)]
#[allow(unused_imports)]
use std::fs;
use std::{env, error::Error, path::Path};

#[derive(Default)]
struct Parameters {
    #[cfg(feature = "spv-in")]
    spv_adjust_coordinate_space: bool,
    #[cfg(feature = "spv-in")]
    spv_flow_dump_prefix: Option<String>,
    #[cfg(feature = "spv-out")]
    spv: naga::back::spv::Options,
    #[cfg(feature = "msl-out")]
    msl: naga::back::msl::Options,
    #[cfg(feature = "glsl-out")]
    glsl: naga::back::glsl::Options,
}

trait PrettyResult {
    type Target;
    fn unwrap_pretty(self) -> Self::Target;
}

fn print_err(error: impl Error) {
    println!("{}:", error);
    let mut e = error.source();
    while let Some(source) = e {
        println!("\t{}", source);
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
    //env_logger::init(); // uncomment during development

    let mut input_path = None;
    let mut output_path = None;
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
                #[cfg(feature = "spv-in")]
                "flow-dir" => params.spv_flow_dump_prefix = args.next(),
                #[cfg(feature = "glsl-out")]
                "entry-point" => params.glsl.entry_point = args.next().unwrap(),
                #[cfg(feature = "glsl-out")]
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
                other => log::warn!("Unknown parameter: {}", other),
            }
        } else if input_path.is_none() {
            input_path = Some(arg);
        } else if output_path.is_none() {
            output_path = Some(arg);
        } else {
            log::warn!("Extra parameter: {}", arg);
        }
    }

    let input_path = match input_path {
        Some(ref string) => string,
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
        #[cfg(feature = "spv-in")]
        "spv" => {
            let options = naga::front::spv::Options {
                adjust_coordinate_space: params.spv_adjust_coordinate_space,
                flow_graph_dump_prefix: params.spv_flow_dump_prefix.map(std::path::PathBuf::from),
            };
            let input = fs::read(input_path).unwrap();
            naga::front::spv::parse_u8_slice(&input, &options).unwrap()
        }
        #[cfg(feature = "wgsl-in")]
        "wgsl" => {
            let input = fs::read_to_string(input_path).unwrap();
            let result = naga::front::wgsl::parse_str(&input);
            match result {
                Ok(v) => v,
                Err(ref e) => {
                    e.emit_to_stderr();
                    panic!("unable to parse WGSL");
                }
            }
        }
        #[cfg(feature = "glsl-in")]
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
            .unwrap_pretty()
        }
        #[cfg(feature = "glsl-in")]
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
            .unwrap_pretty()
        }
        #[cfg(feature = "glsl-in")]
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
            .unwrap_pretty()
        }
        other => {
            if true {
                // prevent "unreachable_code" warnings
                panic!("Unknown input extension: {}", other);
            }
            naga::Module::default()
        }
    };

    // validate the IR
    #[allow(unused_variables)]
    let info =
        match naga::valid::Validator::new(naga::valid::ValidationFlags::all()).validate(&module) {
            Ok(info) => Some(info),
            Err(error) => {
                print_err(error);
                None
            }
        };

    let output_path = match output_path {
        Some(ref string) => string,
        None => {
            println!("{:#?}", module);
            println!("{:#?}", info);
            return;
        }
    };

    match Path::new(output_path)
        .extension()
        .expect("Output has no extension?")
        .to_str()
        .unwrap()
    {
        #[cfg(feature = "msl-out")]
        "metal" => {
            use naga::back::msl;
            let (msl, _) =
                msl::write_string(&module, info.as_ref().unwrap(), &params.msl).unwrap_pretty();
            fs::write(output_path, msl).unwrap();
        }
        #[cfg(feature = "spv-out")]
        "spv" => {
            use naga::back::spv;

            let spv = spv::write_vec(&module, info.as_ref().unwrap(), &params.spv).unwrap_pretty();
            let bytes = spv
                .iter()
                .fold(Vec::with_capacity(spv.len() * 4), |mut v, w| {
                    v.extend_from_slice(&w.to_le_bytes());
                    v
                });

            fs::write(output_path, bytes.as_slice()).unwrap();
        }
        #[cfg(feature = "glsl-out")]
        stage @ "vert" | stage @ "frag" | stage @ "comp" => {
            use naga::back::glsl;

            params.glsl.shader_stage = match stage {
                "vert" => naga::ShaderStage::Vertex,
                "frag" => naga::ShaderStage::Fragment,
                "comp" => naga::ShaderStage::Compute,
                _ => unreachable!(),
            };

            let file = fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(output_path)
                .unwrap();

            let mut writer = glsl::Writer::new(file, &module, info.as_ref().unwrap(), &params.glsl)
                .unwrap_pretty();

            writer
                .write()
                .map_err(|e| {
                    fs::remove_file(output_path).unwrap();
                    e
                })
                .unwrap();
        }
        #[cfg(feature = "dot-out")]
        "dot" => {
            use naga::back::dot;
            let output = dot::write(&module, info.as_ref()).unwrap();
            fs::write(output_path, output).unwrap();
        }
        other => {
            let _ = params;
            panic!(
                "Unknown output extension: {}, forgot to enable a feature?",
                other
            );
        }
    }
}
