use serde::{Deserialize, Serialize};
use std::{env, fs, path::Path};

#[derive(Hash, PartialEq, Eq, Serialize, Deserialize)]
enum Stage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Hash, PartialEq, Eq, Serialize, Deserialize)]
struct BindSource {
    stage: Stage,
    group: u32,
    binding: u32,
}

#[derive(Serialize, Deserialize)]
struct BindTarget {
    #[serde(default)]
    buffer: Option<u8>,
    #[serde(default)]
    texture: Option<u8>,
    #[serde(default)]
    sampler: Option<u8>,
    #[serde(default)]
    mutable: bool,
}

#[derive(Default, Serialize, Deserialize)]
struct Parameters {
    #[serde(default)]
    spv_flow_dump_prefix: String,
    metal_bindings: naga::FastHashMap<BindSource, BindTarget>,
}

fn main() {
    env_logger::init();

    let args = env::args().collect::<Vec<_>>();

    if args.len() <= 1 {
        println!("Call with <input> <output>");
        return;
    }

    let param_path = std::path::PathBuf::from(&args[1]).with_extension("param.ron");
    let params = match fs::read_to_string(param_path) {
        Ok(string) => ron::de::from_str(&string).unwrap(),
        Err(_) => Parameters::default(),
    };

    let module = match Path::new(&args[1])
        .extension()
        .expect("Input has no extension?")
        .to_str()
        .unwrap()
    {
        #[cfg(feature = "spv-in")]
        "spv" => {
            let options = naga::front::spv::Options {
                flow_graph_dump_prefix: if params.spv_flow_dump_prefix.is_empty() {
                    None
                } else {
                    Some(params.spv_flow_dump_prefix.into())
                },
            };
            let input = fs::read(&args[1]).unwrap();
            naga::front::spv::parse_u8_slice(&input, &options).unwrap()
        }
        #[cfg(feature = "wgsl-in")]
        "wgsl" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::wgsl::parse_str(&input).unwrap()
        }
        #[cfg(feature = "glsl-in")]
        "vert" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::glsl::parse_str(
                &input,
                "main",
                naga::ShaderStage::Vertex,
                Default::default(),
            )
            .unwrap()
        }
        #[cfg(feature = "glsl-in")]
        "frag" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::glsl::parse_str(
                &input,
                "main",
                naga::ShaderStage::Fragment,
                Default::default(),
            )
            .unwrap()
        }
        #[cfg(feature = "glsl-in")]
        "comp" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::glsl::parse_str(
                &input,
                "main",
                naga::ShaderStage::Compute,
                Default::default(),
            )
            .unwrap()
        }
        #[cfg(feature = "deserialize")]
        "ron" => {
            let mut input = fs::File::open(&args[1]).unwrap();
            ron::de::from_reader(&mut input).unwrap()
        }
        other => {
            if true {
                // prevent "unreachable_code" warnings
                panic!("Unknown input extension: {}", other);
            }
            naga::Module::generate_empty()
        }
    };

    if args.len() <= 2 {
        println!("{:#?}", module);
        return;
    }

    match Path::new(&args[2])
        .extension()
        .expect("Output has no extension?")
        .to_str()
        .unwrap()
    {
        #[cfg(feature = "msl-out")]
        "metal" => {
            use naga::back::msl;
            let mut binding_map = msl::BindingMap::default();
            for (key, value) in params.metal_bindings {
                binding_map.insert(
                    msl::BindSource {
                        stage: match key.stage {
                            Stage::Vertex => naga::ShaderStage::Vertex,
                            Stage::Fragment => naga::ShaderStage::Fragment,
                            Stage::Compute => naga::ShaderStage::Compute,
                        },
                        group: key.group,
                        binding: key.binding,
                    },
                    msl::BindTarget {
                        buffer: value.buffer,
                        texture: value.texture,
                        sampler: value.sampler,
                        mutable: value.mutable,
                    },
                );
            }
            let options = msl::Options {
                lang_version: (1, 0),
                spirv_cross_compatibility: false,
                binding_map,
            };
            let msl = msl::write_string(&module, &options).unwrap();
            fs::write(&args[2], msl).unwrap();
        }
        #[cfg(feature = "spv-out")]
        "spv" => {
            use naga::back::spv;

            let debug_flag = args.get(3).map_or(spv::WriterFlags::DEBUG, |arg| {
                if arg.parse().unwrap() {
                    spv::WriterFlags::DEBUG
                } else {
                    spv::WriterFlags::NONE
                }
            });

            let spv = spv::Writer::new(&module.header, debug_flag).write(&module);

            let bytes = spv
                .iter()
                .fold(Vec::with_capacity(spv.len() * 4), |mut v, w| {
                    v.extend_from_slice(&w.to_le_bytes());
                    v
                });

            fs::write(&args[2], bytes.as_slice()).unwrap();
        }
        #[cfg(feature = "glsl-out")]
        stage @ "vert" | stage @ "frag" | stage @ "comp" => {
            use naga::{
                back::glsl::{self, Options, Version},
                ShaderStage,
            };

            let version = match args.get(3).map(|p| p.as_str()) {
                Some("core") => {
                    Version::Desktop(args.get(4).and_then(|v| v.parse().ok()).unwrap_or(330))
                }
                Some("es") => {
                    Version::Embedded(args.get(4).and_then(|v| v.parse().ok()).unwrap_or(310))
                }
                Some(_) => panic!("Unknown profile"),
                _ => Version::Embedded(310),
            };

            let options = Options {
                version,
                entry_point: (
                    match stage {
                        "vert" => ShaderStage::Vertex,
                        "frag" => ShaderStage::Fragment,
                        "comp" => ShaderStage::Compute,
                        _ => unreachable!(),
                    },
                    String::from("main"),
                ),
            };

            let file = fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&args[2])
                .unwrap();

            let mut writer = glsl::Writer::new(file, &module, &options).unwrap();

            writer
                .write()
                .map_err(|e| {
                    fs::remove_file(&args[2]).unwrap();
                    e
                })
                .unwrap();
        }
        #[cfg(feature = "serialize")]
        "ron" => {
            let config = ron::ser::PrettyConfig::new()
                .with_enumerate_arrays(true)
                .with_decimal_floats(true);

            let output = ron::ser::to_string_pretty(&module, config).unwrap();
            fs::write(&args[2], output).unwrap();
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
